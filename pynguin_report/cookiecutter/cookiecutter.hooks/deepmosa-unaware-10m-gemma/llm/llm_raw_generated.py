####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize("hook_exception, delete_expected", [
    (FailedHookException("failed"), True),
    (UndefinedError("undefined"), False),
    (RuntimeError("unexpected"), True), # Testing the catch-all logic via exception propagation
])
def test_run_hook_from_repo_dir(hook_exception, delete_expected):
    """Test run_hook_from_repo_dir handles exceptions and directory deletion."""
    # Setup mocks
    repo_dir = "/fake/repo"
    project_dir = "/fake/project"
    hook_name = "post_gen_project"
    context = {"foo": "bar"}
    delete_project_on_failure = True

    with patch("hooks.work_in") as mock_work_in, \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        # Configure the mock to raise the specific exception when run_hook is called
        mock_run_hook.side_effect = hook_exception

        # Execute function under test
        with pytest.raises((FailedHookException, UndefinedError, RuntimeError)):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure
            )

        # Verify work_in was called with repo_dir
        mock_work_in.assert_called_with(repo_dir)

        # Verify run_hook was called with correct arguments
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

        # Verify rmtree is called only if delete_project_on_failure is True and exception occurs
        if delete_expected:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_success():
    """Test run_hook_from_repo_dir when the hook runs successfully."""
    repo_dir = "/fake/repo"
    project_dir = "/fake/project"
    hook_name = "post_gen_project"
    context = {"foo": "bar"}

    with patch("hooks.work_in") as mock_work_in, \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        # Hook succeeds (returns None)
        mock_run_hook.return_value = None

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )

        mock_work_in.assert_called_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize(
    "hook_name, project_dir, context, scripts, expected_call",
    [
        (
            "post_gen_project",
            "/tmp/project",
            {"name": "test"},
            ["/tmp/hooks/post_gen_project.py"],
            True,
        ),
        (
            "post_gen_project",
            "/tmp/project",
            {"name": "test"},
            [],
            False,
        ),
    ],
)
def test_run_hook(hook_name, project_dir, context, scripts, expected_call):
    """Test run_hook correctly identifies and executes hooks."""
    with patch("cookiecutter.hooks.find_hook") as mock_find:
        with patch("cookiecutter.hooks.run_script_with_context") as mock_run_ctx:
            mock_find.return_value = scripts

            run_hook(hook_name, project_dir, context)

            if expected_call:
                # Verify run_script_with_context was called for each script found
                assert mock_run_ctx.call_count == len(scripts)
                for script in scripts:
                    mock_run_ctx.assert_any_call(script, project_dir, context)
            else:
                # Verify nothing was run if no hooks were found
                mock_run_ctx.assert_not_called()

    # Verify find_hook was called with the correct hook name
    mock_find.assert_called_once_with(hook_name)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("hook_name, delete_project_on_failure, exception_to_raise", [
    ("pre_gen_project", True, FailedHookException),
    ("pre_gen_project", True, UndefinedError),
    ("post_gen_project", False, FailedHookException),
    ("post_gen_project", False, UndefinedError),
])
def test_run_hook_from_repo_dir(hook_name, delete_project_on_failure, exception_to_raise):
    """
    Test run_hook_from_repo_dir handles success and failure scenarios,
    specifically checking if project directory is deleted on failure.
    """
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.utils.work_in"), \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiejack.hooks.rmtree") as mock_rmtree:
        
        # Setup the failure injection
        mock_run_hook.side_effect = exception_to_raise("Hook failed")

        with pytest.raises(exception_to_raise):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure,
            )

        # Verify run_hook was called with correct args
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

        # Verify rmtree is only called if delete_project_on_failure is True
        if delete_project_on_failure:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_success():
    """Test run_hook_from_repo_dir success path."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.utils.work_in"), \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # Success scenario
        mock_run_hook.return_value = None

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )

        mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook(tmp_path):
    """Test the find_hook function with various scenarios."""
    
    # Scenario 1: hooks directory does not exist
    non_existent_dir = tmp_path / "no_hooks"
    assert find_hook("pre_gen_project", str(non_existent_dir)) is None

    # Scenario 2: hooks directory exists but is empty
    empty_hooks_dir = tmp_path / "empty_hooks"
    empty_hooks_dir.mkdir()
    assert find_hook("pre_gen_project", str(empty_hooks_dir)) is None

    # Scenario 3: hooks directory exists with valid hooks
    hooks_dir = tmp_path / "valid_hooks"
    hooks_dir.mkdir()
    
    # Create a valid hook file
    valid_hook_file = hooks_dir / "pre_gen_project.py"
    valid_hook_file.write_text("#!/bin/bash\necho 'hello'")
    
    # Create an invalid hook name file (unsupported name)
    invalid_name_file = hooks_dir / "unknown_hook.py"
    invalid_name_file.write_text("#!/bin/bash\necho 'hello'")
    
    # Create a backup file (should be ignored by valid_hook)
    backup_hook_file = hooks_dir / "post_gen_project.py~"
    backup_hook_file.write_text("#!/bin/bash\necho 'hello'")

    # We must use work_in or change cwd because find_hook relies on os.listdir(hooks_dir) 
    # but the logic uses relative paths for finding scripts based on hooks_dir arg.
    # However, find_hook uses os.path.join(hooks_dir, ...) so it works with absolute paths.
    
    # Test finding a valid hook
    scripts = find_hook("pre_gen_project", str(hooks_dir))
    assert scripts is not None
    assert len(scripts) == 1
    assert os.path.abspath(str(valid_hook_file)) in scripts

    # Test finding a different valid hook (if we added one)
    post_gen_file = hooks_dir / "post_gen_project.py"
    post_gen_file.write_text("#!/bin/bash\necho 'hello'")
    scripts = find_hook("post_gen_project", str(hooks_dir))
    assert len(scripts) == 1
    assert os.path.abspath(str(post_gen_file)) in scripts

    # Test finding a hook name that doesn't exist in the dir
    assert find_hook("pre_prompt", str(hooks_dir)) is not None # should find pre_gen and post_gen if they were there, 
                                                              # but since we only check for existence of matches:
    # Let's verify specifically that invalid names/backups are filtered out
    # Only pre_gen_project.py (valid) and post_gen_project.py (added just now) should be found.
    # The backup file ends in ~ so it is excluded by valid_hook.
    # The unknown_hook.py is excluded because its name isn't in _HOOKS.
    
    scripts = find_hook("pre_gen_project", str(hooks_dir))
    for s in scripts:
        basename = os.path.basename(s)
        assert not basename.endswith('~')
        assert os.path.splitext(basename)[0] == "pre_gen_project"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize("hook_name, delete_project_on_failure, exception_to_raise", [
    ("pre_gen_project", True, FailedHookException),
    ("pre_gen_project", False, FailedHookException),
    ("post_gen_project", True, UndefinedError),
])
def test_run_hook_from_repo_dir(hook_name, delete_project_on_failure, exception_to_raise):
    """Test run_hook_from_repo_dir handles success and failure scenarios."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}

    with patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.work_in"), \
         patch("cookielama.hooks.rmtree") as mock_rmtree:
        
        # Setup the exception to be raised by run_hook
        mock_run_hook.side_effect = exception_to_raise("Hook failed")

        with pytest.raises(exception_to_raise):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=delete_project_on_failure
            )

        # Verify run_hook was called with correct arguments
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

        # If failure occurred and delete flag is True, rmtree should be called on project_dir
        if delete_project_on_failure:
            mock_rmtree.assert_called_once_with(project_dir)
        else:
            mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_success():
    """Test run_hook_from_repo_dir success scenario."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}

    with patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.work_in"), \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # No exception raised means success
        mock_run_hook.return_value = None

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )

        mock_run_hook.assert_called_once_with("post_gen_project", project_dir, context)
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@patch("os.path.splitext")
@patch("pathlib.Path.read_text")
@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_create_env,
    mock_tempfile,
    mock_read_text,
    mock_splitext
):
    # Setup inputs
    script_path = "/tmp/hooks/pre_gen_project.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    script_content = "Hello {{ project_name }}"
    rendered_content = "Hello test_project"
    extension = ".py"

    # Mocking file system behavior
    mock_splitext.return_value = (script_path, extension)
    mock_read_text.return_value = script_content
    
    # Mocking Jinja2 environment and rendering
    mock_template = MagicMock()
    mock_template.render.return_value = rendered_content
    mock_env = MagicMock()
    mock_env.from_string.return_value = mock_template
    mock_create_env.return_value = mock_env

    # Mocking NamedTemporaryFile context manager
    mock_temp_instance = MagicMock()
    mock_tempfile.return_value.__enter__.return_value = mock_temp_instance
    mock_temp_instance.name = "/tmp/temp_script.py"

    # Execute the function
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(script_path, cwd, context)

    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(context)
    mock_env.from_string.assert_called_once_with(script_content)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify the content was written to the temp file encoded as bytes
    mock_temp_instance.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Verify run_script was called with the temporary file path and correct cwd
    mock_run_script.assert_called_once_with(mock_temp_instance.name, cwd)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook(mock_run_with_context, mock_find_hook):
    """Test that run_hook finds and executes scripts correctly."""
    
    # Define common inputs
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter_project_slug': 'my_project'}

    # Case 1: No hooks found
    mock_find_hook.return_value = None
    run_hook(hook_name, project_dir, context)
    
    mock_find_hook.assert_called_with(hook_name)
    mock_run_with_context.assert_not_called()

    # Case 2: One hook found and executed
    mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.py']
    run_hook(hook_name, project_dir, context)
    
    mock_run_with_context.assert_called_once_with(
        '/tmp/hooks/pre_gen_project.py', 
        project_dir, 
        context
    )

    # Case 3: Multiple hooks found and executed sequentially
    mock_find_hook.return_value = [
        '/tmp/hooks/pre_gen_project.py',
        '/tmp/hooks/pre_gen_project.sh'
    ]
    # Reset mock to clear previous call history for clean assertion
    mock_run_with_context.reset_mock()
    
    run_hook(hook_name, project_dir, context)
    
    assert mock_run_with_context.call_count == 2
    mock_run_with_context.assert_any_call('/tmp/hooks/pre_gen_project.py', project_dir, context)
    mock_run_with_context.assert_any_call('/tmp/hooks/pre_gen_project.sh', project_dir, context)

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook_exception_propagation(mock_run_with_context, mock_find_hook):
    """Test that run_hook propagates exceptions from the execution step."""
    from cookiecutter.exceptions import FailedHookException

    hook_name = 'post_gen_project'
    project_dir = '/tmp/project'
    context = {}
    
    mock_find_hook.return_value = ['/tmp/hooks/post_gen_project.py']
    mock_run_with_context.side_effect = FailedHookException("Script failed")

    with pytest.raises(FailedHookException, match="Script failed"):
        run_hook(hook_name, project_dir, context)
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def temp_repo_dir(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook(temp_repo_dir):
    """Test run_pre_prompt_hook functionality."""
    
    # Case 1: No pre_prompt hook exists in the repo_dir
    # The function should return the original repo_dir unchanged
    with patch("cookiecutter.hooks.hooks.work_in"):
        with patch("cookiecutter.hooks.hooks.find_hook", return_value=None):
            result = run_pre_prompt_hook(str(temp_repo_dir))
            assert result == str(temp_repo_dir)

    # Case 2: pre_prompt hook exists and runs successfully
    # We mock find_hook to return a script path, and run_script to succeed
    script_path = str(temp_repo_dir / "hooks" / "pre_prompt.py")
    
    with patch("cookiecutter.hooks.hooks.work_in"):
        with patch("cookiecutter.hooks.hooks.find_hook", return_value=[script_path]):
            with patch("cookiecutter.hooks.hooks.create_tmp_repo_dir", return_value=str(temp_repo_dir)):
                with patch("cookiecutter.hooks.hooks.run_script") as mock_run:
                    result = run_pre_prompt_hook(str(temp_repo_dir))
                    
                    # Verify the script was executed with the correct cwd
                    mock_run.assert_called_once_with(script_path, str(temp_repo_dir))
                    assert result == str(temp_repo_dir)

    # Case 3: pre_prompt hook exists but fails
    # The function should catch FailedHookException and re-raise a new one with specific message
    with patch("cookiecutter.hooks.hooks.work_in"):
        with patch("cookiecutter.hooks.hooks.find_hook", return_value=[script_path]):
            with patch("cookiecutter.hooks.hooks.create_tmp_repo_dir", return_value=str(temp_repo_dir)):
                with patch("cookiecutter.hooks.hooks.run_script", side_effect=FailedHookException("Original error")):
                    with pytest.raises(FailedHookException) as excinfo:
                        run_pre_prompt_hook(str(temp_repo_dir))
                    
                    assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize("hook_type, should_delete", [
    ("success", False),
    ("failure_failed_hook", True),
    ("failure_undefined_error", True),
])
def test_run_hook_from_repo_dir(hook_type, should_delete):
    """
    Tests run_hook_from_repo_dir for various execution scenarios.
    """
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "post_gen_project"
    context = {"name": "test"}

    # Mocking the dependencies
    with patch("hooks.work_in") as mock_work_in, \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        # Setup behavior based on hook_type
        if hook_type == "success":
            mock_run_hook.return_value = None
        elif hook_type == "failure_failed_hook":
            mock_run_hook.side_effect = FailedHookException("Failed")
        elif hook_type == "failure_undefined_error":
            mock_run_hook.side_effect = UndefinedError("Undefined")

        # Execute the function under test
        if hook_type == "success":
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=should_delete
            )
            # Assertions for success
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
            mock_rmtree.assert_not_called()

        else:
            # Assertions for failures
            with pytest.raises((FailedHookException, UndefinedError)):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=should_delete
                )
            
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
            if should_delete:
                mock_rmtree.assert_called_once_with(project_dir)
            else:
                mock_rmtree.assert_not_called()

        # Ensure work_in was used as a context manager
        mock_work_in.assert_called_once_with(repo_dir)
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("scripts_found, expected_return", [
    ([], "original_dir"),  # No scripts found, return original
    (["/path/to/pre_prompt.py"], "/tmp/new_dir"),  # Scripts found, return tmp dir
])
def test_run_pre_prompt_hook(scripts_found, expected_return):
    repo_dir = "/original_dir"
    
    with patch("cookiecutter.hooks.find_hook") as mock_find_hook, \
         patch("cookiecutter.hooks.work_in"), \
         patch("cookiecutter.hooks.create_tmp_repo_dir") as mock_create_tmp, \
         patch("cookiecutter.hooks.run_script") as mock_run_script:
        
        # Setup behavior for first find_hook call (checking existence)
        # and second find_hook call (inside the tmp dir)
        mock_find_hook.side_effect = [scripts_found, scripts_found]
        mock_create_tmp.return_value = "/tmp/new_dir"

        result = run_pre_prompt_hook(repo_dir)

        assert result == expected_return
        if scripts_found:
            mock_run_script.assert_called()
        else:
            mock_run_script.assert_not_called()

def test_run_pre_prompt_hook_failure():
    repo_dir = "/original_dir"
    
    with patch("cookiecutter.hooks.find_hook") as mock_find_hook, \
         patch("cookiecutter.hooks.work_in"), \
         patch("cookiecutter.hooks.create_tmp_repo_dir") as mock_create_tmp, \
         patch("cookiecutter.hooks.run_script") as mock_run_script:
        
        # Simulate finding a script but it failing during execution
        mock_find_hook.side_effect = [["/path/to/pre_prompt.py"], ["/path/to/pre_prompt.py"]]
        mock_create_tmp.return_value = "/tmp/new_dir"
        mock_run_script.side_effect = FailedHookException("Script failed")

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook(repo_dir)
        
        assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook(mock_run_script_with_context, mock_find_hook):
    """Test that run_hook finds and executes scripts correctly."""
    # Setup
    hook_name = 'post_gen_project'
    project_dir = '/tmp/project'
    context = {'project_name': 'test_project'}
    scripts = ['/absolute/path/to/post_gen_project.py']

    # Case 1: No hooks found
    mock_find_hook.return_value = None
    run_hook(hook_name, project_dir, context)
    
    mock_find_hook.assert_called_with(hook_name)
    mock_run_script_with_context.assert_not_called()

    # Case 2: Hooks found and executed
    mock_find_hook.return_value = scripts
    run_hook(hook_name, project_dir, context)

    # Verify find_hook was called with correct name
    mock_find_hook.assert_called_with(hook_name)
    # Verify run_script_with_context was called for each script found
    mock_run_script_with_context.assert_called_once_with(
        scripts[0], project_dir, context
    )

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook_multiple_scripts(mock_run_script_with_context, mock_find_hook):
    """Test that run_hook executes all found scripts."""
    hook_name = 'post_gen_project'
    project_dir = '/tmp/project'
    context = {}
    scripts = ['/path/1.py', '/path/2.py']

    mock_find_hook.return_value = scripts

    run_hook(hook_name, project_dir, context)

    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call('/path/1.py', project_dir, context)
    mock_run_script_with_context.assert_any_call('/path/2.py', project_dir, context)
```


# LLM-generated content at query #12
#--------------------------

```python
import os
from unittest.mock import patch, MagicMock
import pytest

def test_find_hook():
    # Test case 1: hooks directory does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Test case 2: hooks directory exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Test case 3: hooks directory exists with non-matching hook files
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['post_gen_project.py', 'other_script.sh']):
        # valid_hook logic: 
        # post_gen_project.py is a supported hook, but we are looking for pre_prompt
        assert find_hook('pre_prompt', 'hooks_dir') is None

    # Test case 4: hooks directory exists with matching hook files
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py', 'post_gen_project.py']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        
        results = find_hook('pre_prompt', 'hooks_dir')
        assert results is not None
        assert len(results) == 1
        assert results[0] == "/abs/path/pre_prompt.py"

    # Test case 5: hooks directory exists with matching hook but it's a backup file (ends with ~)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py~']):
        assert find_hook('pre_prompt', 'hooks_dir') is None

    # Test case 6: hooks directory exists with multiple valid matching scripts
    # (Note: in reality, filesystem collisions might happen, but test logic follows listdir)
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py', 'pre_prompt.sh']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        
        results = find_hook('pre_prompt', 'hooks_dir')
        assert len(results) == 2
        assert "/abs/path/pre_prompt.py" in results
        assert "/abs/path/pre_prompt.sh" in results
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

@pytest.fixture
def temp_script_file(tmp_path):
    """Fixture to create a template script file."""
    script = tmp_path / "pre_gen_project.py"
    content = "print('Hello {{ name }}')"
    script.write_text(content, encoding='utf-8')
    return script

@pytest.fixture
def context():
    """Fixture for template context."""
    return {"name": "world"}

def test_run_script_with_context(temp_script_file, context, tmp_path):
    """
    Tests run_script_with_context by verifying:
    1. The Jinja2 rendering occurs correctly.
    2. A temporary file is created with the rendered content.
    3. run_script is called with the path to the temp file.
    """
    cwd = str(tmp_path)
    
    # We patch 'run_script' to avoid actual execution of a subprocess
    # and patch 'create_env_with_context' to control the Jinja environment
    with patch('cookiecutter.hooks.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.hooks.create_env_with_context') as mock_create_env:
        
        # Setup Mock Environment and Template
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "Hello world"
        
        # Execute the function
        run_script_with_context(temp_script_file, cwd, context)
        
        # Assertions
        # 1. Check if Jinja was called with correct content from file
        mock_env.from_string.assert_called_once()
        # Verify the content passed to from_string matches original file content
        args, _ = mock_env.from_string.call_args
        assert args[0] == "print('Hello {{ name }}')"
        
        # 2. Check if render was called with context
        mock_template.render.assert_called_once_with(**context)
        
        # 3. Check if run_script was called
        # We can't easily predict the exact temp filename, but we can check that it was called
        # and that it points to a file that exists (or at least ends with .py)
        assert mock_run_script.called
        args, kwargs = mock_run_script.call_args
        script_path_arg = args[0]
        assert os.path.exists(script_path_arg)
        assert script_path_arg.endswith('.py')
        
        # 4. Verify the content of the created temp file matches the rendered output
        with open(script_path_arg, 'r', encoding='utf-8') as f:
            assert f.read() == "Hello world"

        # 5. Check if cwd was passed correctly
        assert kwargs['cwd'] == cwd
```


# LLM-generated content at query #14
#--------------------------

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_run_script_with_context(tmp_path):
    """Test run_script_with_context renders template and calls run_script."""
    # Setup: Create a dummy script with Jinja2 syntax
    script_content = "Hello {{ name }}!"
    script_file = tmp_path / "post_gen_project.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {"name": "World"}
    cwd = str(tmp_path)

    # We patch run_script to avoid actual execution and check if it's called correctly
    # We also patch create_env_with_context to control the Jinja environment
    with patch('cookiecutter.hooks.run_script.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.run_script.create_env_with_context') as mock_create_env:
        
        # Mocking the Jinja2 environment and template behavior
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "Hello World!"

        # Execute the function
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_file, cwd, context)

        # Verify: Check if template was rendered with correct context
        mock_template.render.assert_called_once_with(**context)
        
        # Verify: Check if run_script was called
        # Since the function uses a NamedTemporaryFile, we can't predict the exact path,
        # but we can check if any call to run_script was made and it contains the rendered content.
        assert mock_run_script.called
        args, _ = mock_run_script.call_args
        
        # The first argument to run_script is the temp file path
        temp_script_path = Path(args[0])
        assert temp_script_path.exists()
        assert temp_script_path.read_text(encoding='utf-8') == "Hello World!"
        assert args[1] == cwd

def test_run_script_with_context_error_handling(tmp_path):
    """Test run_script_with_context behavior when file reading fails."""
    from cookiecutter.hooks import run_script_with_context
    
    non_existent_file = tmp_path / "missing.py"
    context = {}
    
    with pytest.raises(FileNotFoundError):
        run_script_with_context(non_existent_file, str(tmp_path), context)
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def mock_env(tmp_path):
    """Fixture to create a dummy repo directory structure."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook(mock_env):
    # Scenario 1: No pre_prompt hook exists
    # Should return the original repo_dir unchanged
    with patch("cookiecutter.utils.work_in"):
        result = run_pre_prompt_hook(str(mock_env))
        assert result == str(mock_env)

    # Scenario 2: pre_prompt hook exists and runs successfully
    hook_script = mock_env / "hooks" / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('hello')", encoding="utf-8")
    
    # Mocking find_hook to return our script and run_script to succeed
    with patch("cookiecutter.hooks.find_hook") as mock_find:
        mock_find.return_value = [str(hook_script)]
        with patch("cookiecutter.hooks.run_script") as mock_run:
            # Mock create_tmp_repo_dir to return a new temp path
            new_tmp_dir = tmp_path / "new_tmp_repo"
            new_tmp_dir.mkdir()
            
            with patch("cookiecutter.hooks.create_tmp_repo_dir", return_value=str(new_tmp_dir)):
                # We need to mock find_hook again for the second call inside the new context
                mock_find.side_effect = [[str(hook_script)], [str(hook_script)]]
                
                result = run_pre_prompt_hook(str(mock_env))
                
                assert result == str(new_tmp_dir)
                mock_run.assert_called_once()

    # Scenario 3: pre_prompt hook exists but fails
    hook_script_fail = mock_env / "hooks" / "pre_prompt_fail.py"
    hook_script_fail.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)", encoding="utf-8")
    
    with patch("cookiecutter.hooks.find_hook") as mock_find:
        mock_find.return_value = [str(hook_script_fail)]
        with patch("cookiecutter.hooks.run_script", side_effect=FailedHookException("Failure")):
            with patch("cookiecutter.hooks.create_tmp_repo_dir", return_value=str(mock_env)):
                with pytest.raises(FailedHookException) as excinfo:
                    run_pre_prompt_hook(str(mock_env))
                assert "Pre-Prompt Hook script failed" in str(excinfo.value)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize(
    "exception_to_raise, should_delete",
    [
        (FailedHookException("Hook failed"), True),
        (UndefinedError("Template error"), True),
        (ValueError("Unexpected error"), False),
    ],
)
def test_run_hook_from_repo_dir(exception_to_raise, should_delete):
    """Test that run_hook_from_repo_dir handles hook failures and deletion correctly."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "post_gen_project"
    context = {"name": "test"}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.side_effect = exception_to_raise

        if exception_to_raise in (FailedHookException, UndefinedError):
            with pytest.raises(type(exception_to_raise)):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=should_delete,
                )
            
            if should_delete:
                mock_rmtree.assert_called_once_with(project_dir)
            else:
                mock_rmtree.assert_not_called()
        else:
            # For unexpected errors (like ValueError), it should re-raise without deleting
            with pytest.raises(ValueError):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=should_delete,
                )
            mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_success():
    """Test that run_hook_from_repo_dir completes successfully when no error occurs."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "post_gen_project"
    context = {"name": "test"}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.return_value = None

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )

        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize("exception_type, delete_expected", [
    (FailedHookException, True),
    (UndefinedError, True),
    (ValueError, False),  # Should not be caught by the specific tuple in run_hook_from_repo_dir
])
def test_run_hook_from_repo_dir(exception_type, delete_expected):
    """
    Tests run_hook_from_repo_dir for correct execution, 
    error handling, and directory cleanup.
    """
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "post_gen_project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.utils.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree, \
         patch("logging.Logger.exception") as mock_log_exception:
        
        # Setup the behavior of run_hook to raise the specific exception
        mock_run_hook.side_effect = exception_type("Test Error")

        # Execute the function
        if exception_type in (FailedHookException, UndefinedError):
            with pytest.raises(exception_type):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name=hook_name,
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
        else:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

        # Verify work_in was called with the correct repo directory
        mock_work_in.assert_called_once_with(repo_dir)

        # Verify run_hook was called with correct arguments
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

        # Verify rmtree was called only if the exception is in the caught list 
        # AND delete_project_on_failure is True
        if exception_type in (FailedHookException, UndefinedError):
            mock_rmtree.assert_called_once_with(project_dir)
            mock_log_exception.assert_called_once()
        else:
            mock_rmtree.assert_not_called()
            mock_log_exception.assert_not_called()

def test_run_hook_from_repo_dir_success():
    """Tests run_hook_from_repo_dir when the hook executes successfully."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "post_gen_project"
    context = {"foo": "bar"}

    with patch("cookiecutter.hooks.utils.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        
        # Setup success scenario
        mock_run_hook.return_value = None

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )

        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #18
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

@patch("cookiecutter.utils.create_env_with_context")
@patch("cookiecutter.hooks.run_script")
@patch("tempfile.NamedTemporaryFile")
def test_run_script_with_context(
    mock_tempfile, 
    mock_run_script, 
    mock_create_env_with_context
):
    # Setup inputs
    script_path = Path("/tmp/hooks/post_gen_project.py")
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    original_content = "print('Hello {{ project_name }}')"
    rendered_content = "print('Hello test_project')"

    # Mocking the script file content reading
    with patch("pathlib.Path.read_text", return_value=original_content):
        
        # Mocking Jinja2 environment and template rendering
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env_with_context.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = rendered_content

        # Mocking the NamedTemporaryFile context manager behavior
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/temp_script.py"
        # Mocking the write method for bytes
        mock_temp_file_instance.write = MagicMock()
        # Ensure the context manager returns the instance
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file_instance

        # Execute the function under test
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)

        # Assertions
        # 1. Verify environment was created with correct context
        mock_create_env_with_context.assert_called_once_with(context)
        
        # 2. Verify template rendering occurred with correct data
        mock_env.from_string.assert_called_once_with(original_content)
        mock_template.render.assert_called_once_with(**context)

        # 3. Verify the rendered content was encoded and written to the temp file
        mock_temp_file_instance.write.assert_called_once_with(rendered_content.encode("utf-8"))

        # 4. Verify run_script was called with the path of the temporary file and correct cwd
        mock_run_script.assert_called_once_with(mock_temp_file_instance.name, cwd)
```


# LLM-generated content at query #19
#--------------------------

```python
import sys
import subprocess
import errno
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException

def test_run_script(tmp_path):
    # Setup temporary script file
    script = tmp_path / "test_hook.py"
    script.write_text("#!/usr/bin/env python\nprint('hello')")
    script_str = str(script)

    # 1. Test successful execution of a Python script
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        # Mock make_executable to avoid permission issues on all OSs
        with patch("cookiecutter.utils.make_executable"):
            run_script(script_str)
            
            # Verify it calls python executable for .py files
            args, kwargs = mock_popen.call_args
            assert args[0][0] == sys.executable
            assert args[0][1] == script_str
            assert kwargs["cwd"] == "."

    # 2. Test successful execution of a non-Python shell script
    shell_script = tmp_path / "test_hook.sh"
    shell_script.write_text("#!/bin/bash\necho 'hello'")
    shell_script_str = str(shell_script)

    with patch("subprocess.Popen") as mock_popen:
        mock_proc = Magicmask = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        
        with patch("cookiecutter.utils.make_executable"):
            run_script(shell_script_str)
            args, kwargs = mock_popen.call_args
            # For non-py scripts, the first arg is the script path itself
            assert args[0][0] == shell_script_str

    # 3. Test failure via non-zero exit status
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1  # Exit code 1
        mock_popen.return_value = mock_proc
        
        with patch("cookiecutter.utils.make_executable"):
            with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
                run_script(script_str)

    # 4. Test failure via OSError (e.g., ENOEXEC - missing shebang/empty file)
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
        
        with patch("cookiecutter.utils.make_executable"):
            with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
                run_script(script_str)

    # 5. Test failure via general OSError
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
        
        with patch("cookiecutter.utils.make_executable"):
            with pytest.raises(FailedHookException, match="Hook script failed \(error:"):
                run_script(script_str)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.mark.parametrize(
    "hook_name, project_dir, context, delete_project_on_failure, side_effect, expected_raises",
    [
        # Case 1: Success - No exception raised
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            None,
            None,
        ),
        # Case 2: FailedHookException - Project directory should be deleted
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            FailedHookException("Failure"),
            FailedHookException,
        ),
        # Case 3: FailedHookException - Project directory should NOT be deleted
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            False,
            FailedHookException("Failure"),
            FailedHookException,
        ),
        # Case 4: UndefinedError (Jinja) - Project directory should be deleted
        (
            "post_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            True,
            UndefinedError("Missing variable"),
            UndefinedError,
        ),
    ],
)
def test_run_hook_from_repo_dir(
    hook_name,
    project_dir,
    context,
    delete_project_on_failure,
    side_effect,
    expected_raises,
):
    """Tests run_hook_from_repo_dir for success, failure with cleanup, and failure without cleanup."""
    
    # Mocking dependencies to isolate the function logic
    with patch("hooks.work_in") as mock_work_in:
        with patch("hooks.run_hook") as mock_run_hook:
            with patch("hooks.rmtree") as mock_rmtree:
                with patch("logging.Logger.exception") as mock_log_exception:
                    
                    # Setup the side effect for run_hook
                    if side_effect:
                        mock_run_hook.side_call = side_effect
                        # Use a wrapper to trigger the exception when called
                        def side_effect_wrapper(*args, **kwargs):
                            raise side_effect
                        mock_run_hook.side_effect = side_effect_wrapper
                    else:
                        mock_run_hook.side_effect = None

                    # Define test inputs
                    repo_dir = "/tmp/repo"

                    # Execute the function
                    if expected_raises:
                        with pytest.raises(expected_raises):
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

                    # Assertions
                    # 1. Check if work_in was called with the correct repo_dir
                    mock_work_in.assert_called_with(repo_dir)
                    
                    # 2. Check if run_hook was called with correct params
                    mock_run_hook.assert_called_once_with(
                        hook_name, project_dir, context
                    )

                    # 3. Check if rmtree was called only when failure occurs AND delete_project_on_failure is True
                    if side_effect and delete_project_on_failure:
                        mock_rmtree.assert_called_once_with(project_dir)
                    else:
                        mock_rmtree.assert_not_called()

                    # 4. Check if logger.exception was called on failure
                    if side_effect:
                        mock_log_exception.assert_called_once()
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("pathlib.Path.read_text")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script, 
    mock_read_text, 
    mock_create_env, 
    mock_tempfile
):
    # Setup
    script_path = "/tmp/hooks/post_gen_project.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    template_content = "print('Hello {{ project_name }}')"
    rendered_content = "print('Hello test_project')"
    
    # Mocking Path.read_text
    mock_read_text.return_value = template_content
    
    # Mocking Jinja2 Environment and Template
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_create_env.return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = rendered_content
    
    # Mocking NamedTemporaryFile context manager
    mock_temp = MagicMock()
    mock_temp.name = "/tmp/tmp_script.py"
    mock_temp.__enter__.return_value = mock_temp
    mock_tempfile.return_value = mock_temp

    # Execution
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(Path(script_path), cwd, context)

    # Assertions
    mock_read_text.assert_called_once()
    mock_create_env.assert_called_once_with(context)
    mock_env.from_string.assert_called_once_with(template_content)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify the rendered content was written to the temp file
    # Note: .encode('utf-8') is called on the string in the source
    mock_temp.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Verify run_script was called with the temp file path and original cwd
    mock_run_script.assert_called_once_with(mock_temp.name, cwd)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
import os

@pytest.mark.parametrize(
    "hook_file, hook_name, expected",
    [
        # Valid hooks
        ("pre_prompt.py", "pre_prompt", True),
        ("post_gen_project.sh", "post_gen_project", True),
        ("pre_gen_project", "pre_gen_project", True),
        
        # Wrong hook name
        ("wrong_hook.py", "pre_prompt", False),
        ("pre_prompt_extra.py", "pre_prompt", False),
        
        # Unsupported hook type
        ("invalid_type.py", "invalid_type", False),
        
        # Backup files (ending in ~)
        ("pre_prompt.py~", "pre_prompt", False),
        ("post_gen_project~", "post_gen_project", False),
        
        # Path variations
        ("/path/to/hooks/pre_prompt.py", "pre_prompt", True),
        ("./pre_gen_project.sh", "pre_gen_project", True),
        
        # Edge cases
        ("", "", False),
        ("pre_prompt.txt", "pre_prompt", True), # valid because basename matches and is in _HOOKS
    ],
)
def test_valid_hook(hook_file, hook_name, expected):
    assert valid_hook(hook_file, hook_name) == expected
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("hook_name, scripts, context, expected_calls", [
    # Case 1: No hooks found (should return early without calling run_script_with_context)
    ('pre_gen_project', [], {'project_name': 'test'}, []),
    
    # Case 2: One valid hook found (should call run_script_with_context once)
    ('post_gen_project', ['/path/to/post_gen_project.py'], {'project_name': 'test'}, [
        ('/path/to/post_gen_project.py', '/tmp/project', {'project_name': 'test'})
    ]),
    
    # Case 3: Multiple valid hooks found (should call run_script_with_context for each)
    ('pre_prompt', ['/path/to/pre_prompt.py', '/path/to/pre_prompt.sh'], {'project_name': 'test'}, [
        ('/path/to/pre_prompt.py', '/tmp/project', {'project_name': 'test'}),
        ('/path/to/pre_prompt.sh', '/tmp/project', {'project_name': 'test'})
    ]),
])
def test_run_hook(hook_name, scripts, context, expected_calls):
    """Test run_hook correctly identifies and executes discovered hooks."""
    project_dir = "/tmp/project"
    
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script_with_context:
        
        mock_find_hook.return_value = scripts
        
        run_hook(hook_name, project_dir, context)
        
        # Verify find_hook was called with correct arguments
        mock_find_hook.assert_called_once_with(hook_name)
        
        # Verify run_script_with_context calls matches expected script paths and args
        if not scripts:
            mock_run_script_with_context.assert_not_called()
        else:
            assert mock_run_script_with_context.call_count == len(scripts)
            for i, call_args in enumerate(expected_calls):
                # Check if the call arguments match our expected tuple
                actual_args = mock_run_script_with_context.call_args_list[i].args
                assert actual_args == call_args
```


# LLM-generated content at query #24
#--------------------------

```python
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("os.path.splitext")
@patch("pathlib.Path.read_text")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_read_text,
    mock_splitext,
    mock_create_env,
    mock_tempfile
):
    # Setup
    script_path = "/tmp/hooks/pre_gen_project.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    template_content = "Hello {{ project_name }}"
    rendered_content = "Hello test_project"
    
    # Mocking splitext to return .py extension
    mock_splitext.return_value = (script_path, ".py")
    
    # Mocking file reading
    mock_read_text.return_value = template_content
    
    # Mocking Jinja2 environment and rendering
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_create_env.return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = rendered_content
    
    # Mocking NamedTemporaryFile context manager
    mock_temp_file_instance = MagicMock()
    mock_tempfile.return_value.__enter__.return_value = mock_temp_file_instance
    mock_temp_file_instance.name = "/tmp/tmp_script.py"
    
    # Execute
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(Path(script_path), cwd, context)

    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(context)
    mock_env.from_string.assert_called_once_with(template_content)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify the written content to temp file
    mock_temp_file_instance.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Verify that run_script was called with the temporary file path and correct cwd
    mock_run_script.assert_called_once_with("/tmp/tmp_script.py", cwd)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("hook_name, delete_on_failure", [
    ("post_gen_project", True),
    ("post_gen_project", False),
])
def test_run_hook_from_repo_dir_success(hook_name, delete_on_failure):
    """Test successful execution of a hook."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook") as mock_run_hook:
        
        from hooks import run_hook_from_repo_dir
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_on_failure)
        
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


@pytest.mark.parametrize("exception_type", [FailedHookException, UndefinedError])
def test_run_hook_from_repo_dir_failure_with_delete(exception_type):
    """Test hook failure triggers rmtree if delete_project_on_failure is True."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook", side_effect=exception_type("Error")), \
         patch("hooks.rmtree") as mock_rmtree:
        
        from hooks import run_hook_from_repo_dir
        
        with pytest.raises(exception_type):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
        
        mock_rmtree.assert_called_once_with(project_dir)


@pytest.mark.parametrize("exception_type", [FailedHookException, UndefinedError])
def test_run_hook_from_repo_dir_failure_no_delete(exception_type):
    """Test hook failure does NOT trigger rmtree if delete_project_on_failure is False."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {"project_name": "test"}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook", side_effect=exception_type("Error")), \
         patch("hooks.rmtree") as mock_rmtree:
        
        from hooks import run_hook_from_repo_dir
        
        with pytest.raises(exception_type):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #2
#--------------------------

```python
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
from cookiecutter.exceptions import FailedHookException

@patch("cookiecutter.hooks.hooks.work_in")
@patch("cookiecutter.hooks.hooks.find_hook")
@patch("cookiecutter.hooks.hooks.create_tmp_repo_dir")
@patch("cookiecutter.hooks.hooks.run_script")
def test_run_pre_prompt_hook(
    mock_run_script,
    mock_create_tmp_repo_dir,
    mock_find_hook,
    mock_work_in
):
    """Test run_pre_prompt_hook with various scenarios."""
    repo_dir = "/fake/repo"
    tmp_repo_dir = "/fake/tmp_repo"

    # Scenario 1: No pre_prompt hook found in original repo_dir
    mock_find_hook.return_value = None
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    mock_work_in.assert_called_with(repo_dir)
    mock_create_tmp_repo_dir.assert_not_called()

    # Scenario 2: Hook found, successful execution
    mock_find_hook.side_effect = [["/fake/repo/hooks/pre_prompt.py"], ["/fake/tmp_repo/hooks/pre_prompt.py"]]
    mock_create_tmp_repo_dir.return_value = tmp_repo_dir
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result == tmp_repo_dir
    # Check if work_in was called for both the original and the temp dir
    assert mock_work_in.call_count == 2
    mock_run_script.assert_called_once_with("/fake/tmp_repo/hooks/pre_prompt.py", tmp_repo_dir)

    # Scenario 3: Hook found, but execution fails
    mock_find_hook.side_effect = [["/fake/repo/hooks/pre_prompt.py"], ["/fake/tmp_repo/hooks/pre_prompt.py"]]
    mock_run_script.side_effect = FailedHookException("Script failed")
    
    with pytest.raises(FailedHookException) as excinfo:
        run_pre_prompt_hook(repo_dir)
    
    assert "Pre-Prompt Hook script failed" in str(excinfo.value)

@patch("cookiecutter.hooks.hooks.work_in")
@patch("cookiecutter.hooks.hooks.find_hook")
def test_run_pre_prompt_hook_empty_scripts_list(mock_find_hook, mock_work_in):
    """Test scenario where find_hook returns an empty list in the temp directory."""
    repo_dir = "/fake/repo"
    tmp_repo_dir = "/fake/tmp_repo"
    
    # First call (original dir) finds a hook, second call (temp dir) finds nothing
    mock_find_hook.side_effect = [["/fake/repo/hooks/pre_prompt.py"], []]
    
    with patch("cookiecutter.hooks.hooks.create_tmp_repo_dir", return_value=tmp_repo_dir):
        result = run_pre_prompt_hook(repo_dir)
        assert result == tmp_repo_dir
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

@patch("os.path.splitext")
@patch("pathlib.Path.read_text")
@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_create_env,
    mock_tempfile,
    mock_read_text,
    mock_splitext
):
    # Setup
    script_path = "/tmp/hooks/pre_gen_project.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    template_content = "Hello {{ project_name }}"
    rendered_content = "Hello test_project"
    
    mock_splitext.return_value = (script_path, ".py")
    mock_read_text.return_value = template_content
    
    # Mock Jinja2 environment and rendering
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_create_env.return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = rendered_content
    
    # Mock Temporary File context manager
    mock_temp_file_instance = MagicMock()
    mock_tempfile.return_value.__enter__.return_value = mock_temp_file_instance
    mock_temp_file_instance.name = "/tmp/tmp_script.py"
    
    # Execute
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(script_path, cwd, context)
    
    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(context)
    mock_env.from_string.assert_called_once_with(template_content)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify binary write of rendered content
    mock_temp_file_instance.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Verify the actual script execution call
    mock_run_script.assert_called_once_with("/tmp/tmp_script.py", cwd)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

@pytest.fixture
def sample_context():
    return {"project_name": "test_project", "author": "tester"}

@pytest.fixture
def template_content():
    return "Hello {{ project_name }} from {{ author }}!"

def test_run_script_with_context(sample_context, template_content):
    """
    Test that run_script_with_context correctly:
    1. Reads the script file.
    2. Renders it using Jinja2 with the provided context.
    3. Writes the rendered content to a temporary file.
    4. Calls run_script with the path to the temporary file and the target cwd.
    """
    # Create a dummy script file on disk to be read as a template
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp_script:
        tmp_script.write(template_content)
        script_path = tmp_script.name
    
    cwd = "/fake/project/dir"
    
    # We need to mock create_env_with_context and run_script
    # because run_script actually tries to execute a process.
    # We also mock Path.read_text to ensure it reads our dummy file.
    
    try:
        with patch('cookiecutter.hooks.hooks.create_env_with_template_engine') as mock_env_factory, \
             patch('cookiecutter.hooks.hooks.run_script') as mock_run_script, \
             patch('pathlib.Path.read_text', return_value=template_content), \
             patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            
            # Setup Mock Jinja Environment
            mock_template = MagicMock()
            mock_template.render.return_value = "Hello test_project from tester!"
            mock_env = MagicMock()
            mock_env.from_string.return_value = mock_template
            mock_env_factory.return_value = mock_env

            # Setup Mock Temporary File behavior
            # We simulate the context manager 'with tempfile.NamedTemporaryFile(...) as temp:'
            mock_temp_instance = MagicMock()
            mock_temp_instance.name = "/tmp/rendered_script.py"
            # Mocking write to prevent errors when writing bytes
            mock_temp_instance.write = MagicMock() 
            # Support the context manager protocol
            mock_temp_file.return_value.__enter__.return_value = mock_temp_instance
            
            # Execute the function under test
            from cookiecutter.hooks.hooks import run_script_with_context
            run_script_with_context(Path(script_path), cwd, sample_context)

            # Assertions
            
            # 1. Verify context was passed to template rendering
            mock_template.render.assert_called_once_with(**sample_context)
            
            # 2. Verify run_script was called with the temp file path and correct cwd
            mock_run_script.assert_called_once_with(mock_temp_instance.name, cwd)
            
            # 3. Verify that output was encoded to bytes before writing (as per code: .encode('utf-8'))
            expected_output = "Hello test_project from tester!".encode('utf-8')
            mock_temp_instance.write.assert_called_once_with(expected_output)

    finally:
        # Cleanup the original dummy script file
        if os.path.exists(script_path):
            os.remove(script_path)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError

@pytest.fixture
def mock_params():
    return {
        "repo_dir": "/tmp/repo",
        "hook_name": "post_gen_project",
        "project_dir": "/tmp/project",
        "context": {"project_name": "test_project"},
        "delete_project_on_failure": True
    }

def test_run_hook_from_repo_dir_success(mock_params):
    """Test that run_hook executes successfully without calling rmtree."""
    with patch("cookiecutter.hooks.hooks.work_in"), \
         patch("cookiecutter.hooks.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.hooks.rmtree") as mock_rmtree:
        
        run_hook_from_repo_dir(**mock_params)
        
        mock_run_hook.assert_called_once_with(
            mock_params["hook_name"], 
            mock_params["project_dir"], 
            mock_params["context"]
        )
        mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_failure_deletes_project(mock_params):
    """Test that rmtree is called when a FailedHookException occurs."""
    with patch("cookiecutter.hooks.hooks.work_in"), \
         patch("cookiecutter.hooks.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.side_effect = FailedHookException("Hook failed")
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(**mock_params)
        
        mock_rmtree.assert_called_once_with(mock_params["project_dir"])

def test_run_hook_from_repo_dir_undefined_error_deletes_project(mock_params):
    """Test that rmtree is called when an UndefinedError occurs."""
    with patch("cookiecutter.hooks.hooks.work_in"), \
         patch("cookiecutter.hooks.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.side_effect = UndefinedError("Undefined variable")
        
        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(**mock_params)
        
        mock_rmtree.assert_called_once_with(mock_params["project_dir"])

def test_run_hook_from_repo_dir_no_delete_on_failure(mock_params):
    """Test that rmtree is NOT called if delete_project_on_failure is False."""
    params = mock_params.copy()
    params["delete_project_on_failure"] = False
    
    with patch("cookiecutter.hooks.hooks.work_in"), \
         patch("cookiecutter.hooks.hooks.run_hook") as mock_run_hook, \
         patch("cookiecutter.hooks.hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.side_effect = FailedHookException("Hook failed")
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(**params)
        
        mock_rmtree.assert_not_called()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize(
    "hook_name, project_dir, context, scripts, expected_call",
    [
        # Case 1: No hooks found - should return early without calling run_script_with_context
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            None,
            False,
        ),
        # Case 2: Hooks found - should call run_script_with_context for each script
        (
            "pre_gen_project",
            "/tmp/project",
            {"project_name": "test"},
            ["/tmp/hooks/pre_gen_project.py", "/tmp/hooks/pre_gen_project.sh"],
            True,
        ),
    ],
)
def test_run_hook(hook_name, project_dir, context, scripts, expected_call):
    """
    Test run_hook logic for finding and executing hooks.
    """
    with patch("cookiecutter.hooks.find_hook") as mock_find_hook:
        with patch("cookiecutter.hooks.run_script_with_context") as mock_run_script:
            # Setup the find_hook return value
            mock_find_hook.return_value = scripts
            
            # Execute the function under test
            run_hook(hook_name, project_dir, context)

            if expected_call:
                # Verify run_script_with_context was called for each script found
                assert mock_run_script.call_count == len(scripts)
                for i, script_path in enumerate(scripts):
                    mock_run_script.assert_any_call(
                        script_path, project_dir, context
                    )
            else:
                # Verify nothing was executed if no scripts were found
                mock_run_script.assert_not_called()

            # Verify find_hook was called with the correct hook name
            mock_find_hook.assert_called_once_with(hook_name)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@patch('cookiecutter.hooks.hooks.find_hook')
@patch('cookiecutter.hooks.hooks.run_script_with_context')
def test_run_hook(mock_run_script_with_context, mock_find_hook):
    """Test run_hook with various scenarios: no hooks found, and executing multiple hooks."""
    
    # Scenario 1: No hooks found
    mock_find_hook.return_value = None
    project_dir = "/tmp/project"
    context = {"project_name": "test"}
    
    run_hook("pre_gen_project", project_dir, context)
    
    # Assert find_hook was called and no script execution was attempted
    mock_find_hook.assert_called_with("pre_gen_project")
    mock_run_script_with_context.assert_not_called()

    # Scenario 2: One hook found and executed
    mock_find_hook.return_value = ["/tmp/hooks/pre_gen_project.py"]
    
    run_hook("pre_gen_project", project_dir, context)
    
    # Assert the script was executed with correct arguments
    mock_run_script_with_context.assert_called_once_with(
        "/tmp/hooks/pre_gen_project.py", 
        project_dir, 
        context
    )

    # Scenario 3: Multiple hooks found and all executed
    scripts = ["/tmp/hooks/post_gen_project.py", "/tmp/hooks/other_hook.py"]
    # Note: 'other_hook' would normally be filtered by find_hook logic, 
    # but we are testing the loop in run_hook specifically.
    mock_find_hook.return_value = scripts
    mock_run_script_with_context.reset_mock()

    run_hook("post_gen_project", project_dir, context)

    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(scripts[0], project_dir, context)
    mock_run_script_with_context.assert_any_call(scripts[1], project_dir, context)

@patch('cookiecutter.hooks.hooks.find_hook')
@patch('cookiecutter.hooks.hooks.run_script_with_context')
def test_run_hook_exception_propagation(mock_run_script_with_context, mock_find_hook):
    """Test that exceptions in run_script_with_context propagate through run_hook."""
    
    mock_find_hook.return_value = ["/tmp/hooks/pre_gen_project.py"]
    mock_run_script_with_context.side_effect = Exception("Script failed")
    
    with pytest.raises(Exception) as excinfo:
        run_hook("pre_gen_project", "/tmp/project", {})
    
    assert str(excinfo.value) == "Script failed"
```


# LLM-generated content at query #8
#--------------------------

```python
import unittest.mock as mock
from pathlib import Path
from cookiecutter.exceptions import FailedHookException

@mock.patch("cookiecutter.hooks.find_hook")
@mock.patch("cookiecutter.hooks.run_script_with_context")
def test_run_hook(mock_run_with_context, mock_find_hook):
    """Test run_hook executes scripts when found and does nothing when not."""
    project_dir = "/tmp/project"
    context = {"project_name": "test_project"}
    hook_name = "post_gen_project"

    # Case 1: No hooks found
    mock_find_hook.return_value = None
    run_hook(hook_name, project_dir, context)
    mock_run_with_context.assert_not_called()

    # Case 2: One hook found
    mock_find_hook.return_value = ["/tmp/hooks/post_gen_project.py"]
    run_hook(hook_name, project_dir, context)
    mock_run_with_context.assert_called_once_with(
        "/tmp/hooks/post_gen_project.py", project_dir, context
    )

    # Case 3: Multiple hooks found (e.g., different extensions)
    mock_find_hook.return_value = [
        "/tmp/hooks/post_gen_project.py",
        "/tmp/hooks/post_gen_project.sh",
    ]
    run_hook(hook_name, project_dir, context)
    assert mock_run_with_context.call_count == 3  # 1 from Case 2 + 2 from Case 3

@mock.patch("cookiecutter.hooks.find_hook")
@mock.patch("cookiecutter.hooks.run_script_with_context")
def test_run_hook_exception_propagation(mock_run_with_context, mock_find_hook):
    """Test that exceptions in run_script_with_context propagate through run_hook."""
    project_dir = "/tmp/project"
    context = {"project_name": "test_project"}
    hook_name = "post_gen_project"
    
    mock_find_hook.return_value = ["/tmp/hooks/post_gen_project.py"]
    mock_run_with_context.side_effect = FailedHookException("Script failed")

    with pytest.raises(FailedHookException, match="Script failed"):
        run_hook(hook_name, project_dir, context)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("script_path, expected_command", [
    ("test_hook.py", [sys.executable, "test_hook.py"]),
    ("test_hook.sh", ["test_hook.sh"]),
])
def test_run_script(script_path, expected_command):
    """Test run_script executes the correct command and handles outcomes."""
    with patch("subprocess.Popen") as mock_popen, \
         patch("cookiecutter.utils.make_executable") as mock_make_exec:
        
        # Case 1: Success execution
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        run_script(script_path, cwd="/tmp/test")

        mock_make_exec.assert_called_once_with(script_path)
        # Check if Popen was called with correct command and shell settings
        args, kwargs = mock_popen.call_args
        assert args[0] == expected_command
        assert kwargs["cwd"] == "/tmp/test"
        if sys.platform.startswith("win"):
            assert kwargs["shell"] is True

        # Case 2: Failure execution (non-zero exit code)
        mock_process.wait.return_value = 1
        with pytest.raises(FailedHookException, match="Hook script failed \(exit status: 1\)"):
            run_script(script_path, cwd="/tmp/test")

        # Case 3: OSError (ENOEXEC - missing shebang)
        err = OSError()
        err.errno = 8  # errno.ENOEXEC
        import errno
        mock_process.wait.side_effect = err
        with pytest.raise(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(script_path, cwd="/tmp/test")

        # Case 4: Generic OSError
        err = OSError()
        err.errno = errno.EACCES # Permission denied
        mock_process.wait.side_effect = err
        with pytest.raises(FailedHookException, match="Hook script failed \(error:"):
            run_script(script_path, cwd="/tmp/test")
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.parametrize("scripts, context, expected_calls", [
    # Case 1: No scripts found (find_hook returns None)
    ([None], {"project_name": "test"}, []),
    
    # Case 2: Scripts found and executed
    (["/path/to/pre_gen_project.py", "/path/to/post_gen_project.sh"], {"project_name": "test"}, [
        "/path/to/pre_gen_project.py",
        "/path/to/post_gen_project.sh"
    ]),
])
def test_run_hook(scripts, context, expected_calls):
    """
    Tests run_hook to ensure it correctly finds hooks and calls 
    run_script_with_context for each found script.
    """
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"

    # Mock find_hook to return our controlled list of scripts
    # If scripts is [None], simulate no hooks found
    mock_scripts = [] if scripts == [None] else scripts

    with patch('cookiecutter.hooks.find_hook') as mock_find, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_ctx:
        
        mock_find.return_value = mock_scripts
        
        run_hook(hook_name, project_dir, context)

        # Verify find_hook was called with correct params
        mock_find.assert_called_once_with(hook_name)

        # Verify run_script_with_context calls
        if not expected_calls:
            mock_run_ctx.assert_not_called()
        else:
            assert mock_run_ctx.call_count == len(expected_calls)
            for i, script_path in enumerate(expected_calls):
                mock_run_ctx.assert_any_call(script_path, project_dir, context)
```


# LLM-generated content at query #11
#--------------------------

```python
import unittest.mock as mock
from pathlib import Path

def test_run_hook(tmp_path):
    """Test run_hook executes scripts found by find_hook."""
    # Setup directories and files
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    script_file = hooks_dir / "post_gen_project.py"
    script_file.write_text("#!/usr/bin/env python\nprint('hello')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"name": "test"}
    hook_name = "post_gen_project"

    # We need to mock find_hook and run_script_with_context 
    # because find_hook relies on the current working directory (CWD)
    # and we don't want to actually change CWD in a way that affects other tests.
    with mock.patch('cookiecutter.hooks.find_hook') as mock_find, \
         mock.patch('cookiecutter.hooks.run_script_with_context') as mock_run_ctx:
        
        # Define what find_hook returns (absolute paths)
        mock_find.return_value = [str(script_file.absolute())]
        
        # Execute the function under test
        from cookiecutter.hooks import run_hook
        run_hook(hook_name, str(project_dir), context)
        
        # Assertions
        mock_find.assert_called_once_with(hook_name)
        mock_run_ctx.assert_called_once_with(
            str(script_file.absolute()), 
            str(project_dir), 
            context
        )

def test_run_hook_no_scripts_found(tmp_path):
    """Test run_hook does nothing if no scripts are found."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {}
    hook_name = "non_existent_hook"

    with mock.patch('cookiecutter.hooks.find_hook') as mock_find, \
         mock.patch('cookiesetter.hooks.run_script_with_context') as mock_run_ctx:
        
        # Simulate no scripts found
        mock_find.return_value = None
        
        from cookiecutter.hooks import run_hook
        run_hook(hook_name, str(project_dir), context)
        
        # Verify find_hook was called but run_script_with_context was NOT
        mock_find.assert_called_once_with(hook_name)
        mock_run_ctx.assert_not_called()
```


# LLM-generated content at query #12
#--------------------------

```python
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("pathlib.Path.read_text")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_read_text,
    mock_create_env,
    mock_tempfile
):
    # Setup
    script_path = "/tmp/hook.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    template_content = "print('{{ project_name }}')"
    rendered_content = "print('test_project')"

    # Mocking the template rendering flow
    mock_read_text.return_value = template_content
    
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = rendered_content
    mock_create_env.return_value = mock_env

    # Mocking the tempfile context manager
    mock_temp_instance = MagicMock()
    mock_tempfile.return_value.__enter__.return_value = mock_temp_instance
    mock_temp_instance.name = "/tmp/temp_script.py"

    # Execute
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(Path(script_path), cwd, context)

    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(context)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify content was written to temp file encoded as bytes
    mock_temp_instance.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Verify the actual script execution was called with the temp file path
    mock_run_script.assert_called_once_with("/tmp/temp_script.py", cwd)
```


# LLM-generated content at query #13
#--------------------------

```python
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@patch("os.path.splitext")
@patch("pathlib.Path.read_text")
@patch("tempfile.NamedTemporaryFile")
@patch("cookiecutter.utils.create_env_with_context")
@patch("cookiecutter.hooks.run_script")
def test_run_script_with_context(
    mock_run_script,
    mock_create_env,
    mock_tempfile,
    mock_read_text,
    mock_splitext
):
    # Setup
    script_path = "/tmp/hooks/post_gen_project.py"
    cwd = "/tmp/project"
    context = {"project_name": "test_project"}
    template_content = "Hello {{ project_name }}"
    rendered_content = "Hello test_project"
    
    # Mocking file extension and content
    mock_splitext.return_value = (script_path, ".py")
    mock_read_text.return_value = template_content
    
    # Mocking Jinja2 environment and rendering
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_create_env.return_value = mock_env
    mock_env.from_string.return_value = mock_template
    mock_template.render.return_value = rendered_content
    
    # Mocking the temporary file context manager
    mock_temp = MagicMock()
    mock_temp.name = "/tmp/tmp_script.py"
    mock_temp.__enter__.return_value = mock_temp
    mock_tempfile.return_value = mock_temp

    # Execution
    from cookiecutter.hooks import run_script_with_context
    run_script_with_context(script_path, cwd, context)

    # Assertions
    mock_read_text.assert_called_once_with(encoding='utf-8')
    mock_create_env.assert_called_once_with(context)
    mock_env.from_string.assert_called_once_with(template_content)
    mock_template.render.assert_called_once_with(**context)
    
    # Verify the rendered content was written to the temp file
    mock_temp.write.assert_called_once_with(rendered_content.encode('utf-8'))
    
    # Ensure run_script was called with the temporary file path and correct cwd
    mock_run_script.assert_called_once_with(mock_temp.name, cwd)
```


# LLM-generated content at query #14
#--------------------------

```python
import os
import pytest
from unittest.mock import patch

def test_find_hook(tmp_path):
    """Test find_hook with various scenarios including missing dir, no hooks, and valid hooks."""
    
    # Scenario 1: hooks_dir does not exist
    non_existent_dir = tmp_path / "non_existent"
    assert find_hook("pre_gen_project", str(non_existent_dir)) is None

    # Scenario 2: hooks_dir exists but contains no valid hooks
    empty_hooks_dir = tmp_path / "empty_hooks"
    empty_hooks_dir.mkdir()
    assert find_hook("pre_gen_project", str(empty_hooks_dir)) is None

    # Scenario 3: hooks_dir exists but contains invalid hook names or backup files
    invalid_hooks_dir = tmp_path / "invalid_hooks"
    invalid_hooks_dir.mkdir()
    (invalid_hooks_dir / "wrong_name.py").touch()
    (invalid_hooks_dir / "pre_gen_project.py~").touch()  # Backup file
    (invalid_hooks_dir / "not_a_hook.sh").touch()
    assert find_hook("pre_gen_project", str(invalid_hooks_dir)) is None

    # Scenario 4: hooks_dir contains valid hook scripts
    valid_hooks_dir = tmp_path / "valid_hooks"
    valid_hooks_dir.mkdir()
    
    # Create a list of valid hook files
    hook_files = [
        "pre_prompt.py",
        "pre_gen_project.sh",
        "post_gen_project.py",
        "unrelated_script.py" # Should be ignored by valid_hook logic in find_hook
    ]
    
    for f in hook_files:
        (valid_hooks_dir / f).touch()

    # Test finding a specific valid hook (pre_prompt)
    # Note: find_hook uses os.path.abspath, so we check if the returned path is correct
    results = find_hook("pre_prompt", str(valid_hooks_dir))
    assert results is not None
    assert len(results) == 1
    assert os.path.basename(results[0]) == "pre_prompt.py"
    assert os.path.isabs(results[0])

    # Test finding another valid hook (post_gen_project)
    results_post = find_hook("post_gen_project", str(valid_hooks_dir))
    assert results_post is not None
    assert len(results_post) == 1
    assert os.path.basename(results_post[0]) == "post_gen_project.py"

    # Test searching for a hook name that doesn't exist in the directory
    results_none = find_hook("non_existent_hook", str(valid_hooks_dir))
    assert results_none is None

    # Scenario 5: Verifying it filters out files not in _HOOKS list
    # Even if filename matches hook_name, it must be in ['pre_prompt', 'pre_gen_project', 'post_gen_project']
    # Let's mock the _HOOKS check via valid_hook if we were testing logic, 
    # but here we rely on the actual implementation.
    # We already tested this in Scenario 3 with "wrong_name.py"
```


# LLM-generated content at query #15
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock

def test_find_hook(tmp_path):
    """Test find_hook functionality for various scenarios."""
    
    # Scenario 1: hooks directory does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks directory exists but is empty
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=[]):
            assert find_hook('pre_gen_project', str(hooks_dir)) is None

    # Scenario 3: hooks directory exists but contains no valid hooks
    # (e.g., wrong name, or backup files with '~')
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['wrong_name.py', 'pre_prompt.py~', 'other.sh']):
            assert find_hook('pre_gen_project', str(hooks_dir)) is None

    # Scenario 4: hooks directory contains a valid hook
    valid_hook_name = 'pre_gen_project'
    valid_hook_file = 'pre_gen_project.py'
    # We use the actual absolute path logic from find_hook
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=[valid_hook_file, 'post_gen_project.sh']):
            result = find_hook(valid_hook_name, str(hooks_dir))
            
            assert result is not None
            assert len(result) == 1
            # Ensure it returns the absolute path
            expected_path = os.path.abspath(os.path.join(str(hooks_dir), valid_hook_file))
            assert result[0] == expected_path

    # Scenario 5: hooks directory contains multiple valid hooks of same type (if logic allowed)
    # Though valid_hook checks basename, if two files had same basename but different extensions,
    # the current implementation would return both.
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['pre_prompt.py', 'pre_prompt.sh']):
            result = find_hook('pre_prompt', str(hooks_dir))
            assert len(result) == 2
            assert os.path.abspath(os.path.join(str(hooks_dir), 'pre_prompt.py')) in result
            assert os.path.abspath(os.path.join(str(hooks_dir), 'pre_prompt.sh')) in result

def test_find_hook_integration(tmp_path):
    """Integration test using real filesystem."""
    repo_dir = tmp_path / "template"
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir(parents=True)
    
    # Create a valid hook file
    hook_file = hooks_dir / "pre_gen_project.py"
    hook_file.write_text("#!/bin/bash\necho 'hello'")
    
    # Create an invalid hook file (backup)
    backup_file = hooks_dir / "post_gen_project.py~"
    backup_file.write_text("echo 'bad'")
    
    # Create a valid hook with different name
    other_hook = hooks_dir / "pre_prompt.sh"
    other_hook.write_text("#!/bin/bash\necho 'hi'")

    # Change CWD to repo_dir as required by find_hook docstring
    original_cwd = os.getcwd()
    os.chdir(repo_dir)
    
    try:
        # Test finding pre_gen_project (should find the .py one)
        res_pre_gen = find_hook('pre_gen_project', 'hooks')
        assert len(res_pre_gen) == 1
        assert res_pre_gen[0] == os.path.abspath(str(hooks_dir / "pre_gen_project.py"))

        # Test finding pre_prompt (should find the .sh one, but NOT the ~ backup)
        res_pre_prompt = find_hook('pre_prompt', 'hooks')
        assert len(res_pre_prompt) == 1
        assert res_pre_prompt[0] == os.path.abspath(str(hooks_dir / "pre_prompt.sh"))
        
        # Test finding a non-existent hook name
        res_none = find_hook('non_existent', 'hooks')
        assert res_none is None
        
    finally:
        os.chdir(original_cwd)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from jinja2.exceptions import UndefinedError
from cookiecutter.exceptions import FailedHookException

@pytest.mark.parametrize("should_fail, delete_expected", [
    (False, False),
    (True, True),
    (True, False),
])
def test_run_hook_from_repo_dir(should_fail, delete_expected):
    """Test run_hook_from_repo_dir with success and failure scenarios."""
    
    # Setup paths and context
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    hook_name = "pre_gen_project"
    context = {"project_name": "test"}
    
    # Mocking dependencies
    with patch("hooks.work_in") as mock_work_in, \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        # Configure behavior for run_hook
        if should_fail:
            mock_run_hook.side_effect = FailedHookException("Failed")
        else:
            mock_run_hook.return_value = None

        # Execute the function under test
        # Note: We use a try/except block because we expect an exception when should_fail is True
        try:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
            error_raised = False
        except FailedHookException:
            error_raised = True

        # Assertions
        mock_work_in.assert_called_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        
        if should_fail:
            assert error_raised is True
            if delete_expected:
                mock_rmtree.assert_called_once_with(project_id if 'project_id' in locals() else project_dir)
            else:
                mock_rmtree.assert_not_called()
        else:
            assert error_raised is False
            mock_rmtree.assert_not_called()

@pytest.mark.parametrize("exception_type", [FailedHookException, UndefinedError])
def test_run_hook_from_repo_dir_raises_undefined_error(exception_type):
    """Test that UndefinedError is also caught and triggers deletion if configured."""
    repo_dir = "/tmp/repo"
    project_dir = "/tmp/project"
    context = {}

    with patch("hooks.work_in"), \
         patch("hooks.run_hook") as mock_run_hook, \
         patch("hooks.rmtree") as mock_rmtree:
        
        mock_run_hook.side_effect = exception_type("Error")

        with pytest.raises(exception_type):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="pre_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
        
        mock_rmtree.assert_called_once_with(project_dir)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook(mock_run_script_with_context, mock_find_hook):
    """
    Test run_hook:
    1. Case where no scripts are found (should return early).
    2. Case where scripts are found (should execute each script with context).
    """
    # Setup data
    hook_name = 'post_gen_project'
    project_dir = '/tmp/project'
    context = {'project_name': 'test_project'}
    scripts = ['/abs/path/to/post_gen_project.py', '/abs/path/to/post_gen_project.sh']

    # --- Test Case 1: No scripts found ---
    mock_find_hook.return_value = None
    
    run_hook(hook_name, project_dir, context)
    
    mock_find_hook.assert_called_with(hook_name)
    mock_run_script_with_context.assert_not_called()

    # --- Test Case 2: Scripts found ---
    mock_find_hook.return_value = scripts
    
    run_hook(hook_name, project_dir, context)
    
    # Verify find_hook was called again for the second scenario
    assert mock_find_hook.call_count == 2
    
    # Verify run_script_with_context was called for each script in the list
    assert mock_run_script_with_context.call_count == 2
    
    # Check specific call arguments for the first script
    mock_run_script_with_context.assert_any_call(scripts[0], project_dir, context)
    # Check specific call arguments for the second script
    mock_run_script_with_context.assert_any_call(scripts[1], project_dir, context)

@patch('cookiecutter.hooks.find_hook')
@patch('cookiecutter.hooks.run_script_with_context')
def test_run_hook_exception_propagation(mock_run_script_with_context, mock_find_hook):
    """
    Test that run_hook propagates exceptions raised during script execution.
    """
    from cookiecutter.exceptions import FailedHookException
    
    hook_name = 'post_gen_project'
    project_dir = '/tmp/project'
    context = {}
    scripts = ['/abs/path/to/error_script.py']
    
    mock_find_hook.return_value = scripts
    mock_run_script_with_context.side_effect = FailedHookException("Script failed")

    with pytest.raises(FailedHookException, match="Script failed"):
        run_hook(hook_name, project_dir, context)
```


# LLM-generated content at query #18
#--------------------------

```python
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

class TestRunHook(unittest.TestCase):
    @patch('cookiecutter.hooks.find_hook')
    @patch('cookiecutter.hooks.run_script_with_context')
    def test_run_hook_no_scripts_found(self, mock_run_with_context, mock_find_hook):
        """Test run_hook when no scripts are found."""
        mock_find_hook.return_value = None
        project_dir = "/tmp/project"
        context = {"name": "test"}
        
        from cookiecutter.hooks import run_hook
        run_hook("pre_gen_project", project_dir, context)
        
        mock_find_hook.assert_called_once_with("pre_gen_project")
        mock_run_with_context.assert_not_called()

    @patch('cookiecutter.hooks.find_hook')
    @patch('cookiecutter.hooks.run_script_with_context')
    def test_run_hook_executes_scripts(self, mock_run_with_context, mock_find_hook):
        """Test run_hook executes all found scripts."""
        mock_scripts = ["/abs/path/to/pre_gen_project.py", "/abs/path/to/pre_gen_project.sh"]
        mock_find_hook.return_value = mock_scripts
        project_dir = "/tmp/project"
        context = {"name": "test"}
        
        from cookiecutter.hooks import run_hook
        run_hook("pre_gen_project", project_dir, context)
        
        mock_find_hook.assert_called_once_with("pre_gen_project")
        self.assertEqual(mock_run_with_context.call_count, 2)
        
        # Verify first call
        mock_run_with_context.assert_any_call(
            mock_scripts[0], project_dir, context
        )
        # Verify second call
        mock_run_with_context.assert_any_call(
            mock_scripts[1], project_dir, context
        )

    @patch('cookiecutter.hooks.find_hook')
    @patch('cookiecutter.hooks.run_script_with_context')
    def test_run_hook_exception_propagation(self, mock_run_with_context, mock_find_hook):
        """Test run_hook propagates exceptions from run_script_with_context."""
        mock_find_hook.return_value = ["/abs/path/to/pre_gen_project.py"]
        mock_run_with_context.side_effect = Exception("Script Error")
        
        from cookiecutter.hooks import run_hook, FailedHookException
        
        with self.assertRaises(Exception) as cm:
            run_hook("pre_gen_project", "/tmp/project", {})
        
        self.assertEqual(str(cm.exception), "Script Error")

def test_run_hook():
    """
    Wrapper for pytest compatibility. 
    The actual logic is contained in the TestRunHook class above.
    """
    pass
```


# LLM-generated content at query #19
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def mock_env(tmp_path):
    """Creates a dummy repo directory with a pre_prompt hook."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("#!/bin/bash\nprint('hello')", encoding="utf-8")
    return tmp_path

def test_run_pre_prompt_hook(mock_env):
    """
    Tests run_pre_prompt_hook for three scenarios:
    1. No hook exists (returns original repo_dir).
    2. Hook exists and runs successfully.
    3. Hook exists but fails (raises FailedHookException).
    """
    
    # Scenario 1: No pre_prompt hook in the directory
    empty_repo = mock_env.parent / "empty_repo"
    empty_repo.mkdir(exist_ok=True)
    assert run_pre_prompt_hook(str(empty_repo)) == str(empty_repo)

    # Scenario 2: Hook exists and runs successfully
    # We patch run_script to avoid actual subprocess execution during tests
    with patch("cookiecutter.hooks.run_script") as mock_run_script:
        result = run_pre_prompt_hook(str(mock_env))
        
        # The function should return a new (tmp) repo_dir because it calls create_tmp_repo_dir
        assert result != str(mock_env)
        assert os.path.exists(result)
        mock_run_script.assert_called()

    # Scenario 3: Hook exists but fails
    with patch("cookiecutter.hooks.find_hook") as mock_find:
        # Force find_hook to return our valid script path from the first scenario
        mock_find.return_value = [str(mock_env / "hooks" / "pre_prompt.py")]
        
        with patch("cookiecutter.hooks.run_script") as mock_run_script:
            # Simulate a failure in the script execution
            from cookiecutter.exceptions import FailedHookException
            mock_run_script.side_effect = FailedHookException("Hook failed")
            
            with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
                run_pre_prompt_hook(str(mock_env))

def test_run_pre_prompt_hook_no_hooks_dir(tmp_path):
    """Tests that it returns repo_dir if hooks directory is missing."""
    repo_dir = tmp_path / "no_hooks_folder"
    repo_dir.mkdir()
    # Since 'hooks' folder doesn't exist, find_hook returns None
    assert run_pre_prompt_hook(str(repo_dir)) == str(repo_dir)
```


# LLM-generated content at query #20
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def mock_context():
    return {"project_name": "test_project"}

@pytest.fixture
def temp_repo(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook(temp_repo, mock_context):
    # Case 1: No pre_prompt hook exists in the repo
    # Should return the original repo_dir without calling run_script
    with patch("cookiecutter.hooks.hooks.find_hook") as mock_find:
        mock_find.return_value = None
        result = run_pre_prompt_hook(str(temp_repo))
        assert result == str(temp_repo)
        mock_find.assert_called_with('pre_prompt')

    # Case 2: Pre-prompt hook exists and runs successfully
    # We need to mock find_hook to return a script path, and run_script to succeed
    # We also mock create_tmp_repo_dir to avoid actual filesystem duplication complexity
    hook_script = temp_repo / "hooks" / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('hello')", encoding='utf-8')
    
    # Create a second dummy directory to act as the "tmp repo dir"
    tmp_repo_dir = temp_repo / "tmp_repo"
    tmp_repo_dir.mkdir()
    
    with patch("cookiecutter.hooks.hooks.find_hook") as mock_find, \
         patch("cookiecutter.hooks.hooks.create_tmp_repo_dir") as mock_create_tmp, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script, \
         patch("cookiecutter.hooks.hooks.work_in"):
        
        # Setup: find_hook returns the script in the new tmp dir
        mock_find.side_effect = [
            [str(hook_script)], # First call (initial check)
            [str(hook_script)]  # Second call (inside work_in tmp_dir)
        ]
        mock_create_tmp.return_value = str(tmp_repo_dir)
        
        result = run_pre_prompt_hook(str(temp_repo))
        
        assert result == str(tmp_repo_dir)
        mock_run_script.assert_called_once_with(str(hook_script), str(tmp_repo_dir))

    # Case 3: Pre-prompt hook exists but fails
    with patch("cookiecutter.hooks.hooks.find_hook") as mock_find, \
         patch("cookiecutter.hooks.hooks.create_tmp_repo_dir") as mock_create_tmp, \
         patch("cookiecutter.hooks.hooks.run_script") as mock_run_script, \
         patch("cookiecutter.hooks.hooks.work_in"):
        
        mock_find.side_effect = [
            [str(hook_script)], 
            [str(hook_script)]
        ]
        mock_create_tmp.return_value = str(tmp_repo_dir)
        mock_run_script.side_effect = FailedHookException("Script failed")
        
        with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
            run_pre_prompt_hook(str(temp_repo))
```


# LLM-generated content at query #21
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock

def test_find_hook():
    """Test find_hook with various scenarios."""
    
    # Scenario 1: hooks directory does not exist
    with patch('os.path.isdir', return_value=False):
        assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Scenario 2: hooks directory exists but is empty
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre_gen_project', 'hooks_dir') is None

    # Scenario 3: hooks directory exists, but no matching hook name found
    # valid_hook logic: basename must match hook_name and be in _HOOKS
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['post_gen_project.py', 'other_script.sh']):
        # Searching for pre_prompt, but only post_gen_project exists
        assert find_hook('pre_prompt', 'hooks_dir') is None

    # Scenario 4: hooks directory exists and valid hook found
    # We mock os.path.abspath to return a predictable string
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py', 'post_gen_project.sh']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        
        results = find_hook('pre_prompt', 'hooks_dir')
        
        assert results is not None
        assert len(results) == 1
        assert results[0] == "/abs/path/pre_prompt.py"

    # Scenario 5: hooks directory exists, valid hook name, but it's a backup file
    # valid_hook returns False if filename ends with '~'
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py~']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        assert find_hook('pre_prompt', 'hooks_dir') is None

    # Scenario 6: Multiple valid hooks found
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre_prompt.py', 'post_gen_project.py']), \
         patch('os.path.abspath', side_effect=lambda x: f"/abs/path/{os.path.basename(x)}"):
        
        # Note: find_hook takes a specific hook_name to search for. 
        # In the original code, valid_hook checks if basename == hook_name.
        # So find_hook only returns scripts matching the requested name.
        results = find_hook('pre_prompt', 'hooks_dir')
        assert len(results) == 1
        assert "/abs/path/pre_prompt.py" in results
```


# LLM-generated content at query #22
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def temp_repo(tmp_path):
    """Fixture to create a temporary repo directory."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    return tmp_path

def test_run_pre_prompt_hook(temp_repo):
    """Test run_pre_prompt_hook with various scenarios."""
    
    # Scenario 1: No pre_prompt hook exists in the repo
    # Should return the original repo_dir
    with patch('cookiecutter.hooks.hooks.work_in'):
        result = run_pre_prompt_hook(str(temp_repo))
        assert result == str(temp_repo)

    # Scenario 2: pre_prompt hook exists and runs successfully
    # Create a dummy valid script
    hooks_dir = temp_repo / "hooks"
    script_path = hooks_dir / "pre_prompt.py"
    script_path.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    
    # Mocking create_tmp_repo_dir to return a new tmp path so we don't pollute original
    new_tmp_dir = temp_repo / "new_tmp_repo"
    new_tmp_dir.mkdir()
    
    with patch('cookiecutter.hooks.hooks.create_tmp_repo_dir', return_value=str(new_tmp_dir)):
        with patch('cookiecutter.hooks.hooks.run_script') as mock_run:
            result = run_pre_prompt_hook(str(temp_repo))
            
            # Verify it returned the new temp directory
            assert result == str(new_tmp_dir)
            # Verify run_script was called with the script path and the new cwd
            mock_run.assert_called()
            args, kwargs = mock_run.call_args
            assert args[0] == str(script_path)
            assert args[1] == str(new_tmp_dir)

    # Scenario 3: pre_prompt hook exists but fails
    with patch('cookiecutter.hooks.hooks.create_tmp_repo_dir', return_value=str(new_tmp_dir)):
        with patch('cookiecutter.hooks.hooks.run_script') as mock_run:
            from cookiecutter.exceptions import FailedHookException
            mock_run.side_effect = FailedHookException("Script failed")
            
            with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
                run_pre_prompt_hook(str(temp_repo))

    # Scenario 4: Verify work_in context manager is used
    with patch('cookiecutter.hooks.hooks.work_in') as mock_work_in:
        # Setup a dummy hook so it enters the second block of logic
        (temp_repo / "hooks" / "pre_prompt.py").write_text("print('hello')")
        run_pre_prompt_hook(str(temp_repo))
        assert mock_work_in.called
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

@pytest.mark.parametrize("scripts_found, script_fails, expected_return_type", [
    ([], False, "repo_dir"),           # No scripts found: returns original repo_dir
    (["/path/to/pre_prompt.py"], False, "tmp_dir"),  # Script exists and succeeds: returns tmp_dir
    (["/path/to/pre_prompt.py"], True, "raise_exception"), # Script exists but fails: raises exception
])
def test_run_pre_prompt_hook(scripts_found, script_fails, expected_return_type):
    repo_dir = "/fake/repo/dir"
    tmp_repo_dir = "/fake/tmp/repo/dir"

    # Setup mocks for finding hooks and running scripts
    with patch("os.path.isdir", return_value=True), \
         patch("cookiecutter.utils.work_in"), \
         patch("cookiesetter.hooks.find_hook") as mock_find_hook, \
         patch("cookiecutter.utils.create_tmp_repo_dir", return_value=tmp_repo_dir), \
         patch("cookiesetter.hooks.run_script") as mock_run_script:

        # Scenario 1: No pre_prompt hook exists in the original repo_dir
        if not scripts_found:
            mock_find_hook.return_value = None
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
            mock_run_script.assert_not_called()
            return

        # Scenario 2 & 3 setup: Hook exists
        mock_find_hook.side_effect = [scripts_found, scripts_found]
        
        if script_fails:
            mock_run_script.side_effect = FailedHookException("Hook failed")
            with pytest.raises(FailedHookException) as excinfo:
                run_pre_prompt_hook(repo_dir)
            assert "Pre-Prompt Hook script failed" in str(excinfo.value)
        else:
            # Successful execution
            result = run_pre_prompt_hook(repo_dir)
            assert result == tmp_repo_dir
            mock_run_script.assert_called()
            # Verify it calls the script with the temp directory as CWD
            args, kwargs = mock_run_script.call_args
            assert args[1] == tmp_repo_dir

def test_run_pre_prompt_hook_empty_scripts_list():
    """Test behavior when find_hook returns an empty list explicitly."""
    repo_dir = "/fake/repo/dir"
    
    with patch("cookiecutter.utils.work_in"), \
         patch("cookiesetter.hooks.find_hook", return_value=[]), \
         patch("cookiecutter.utils.create_tmp_repo_dir", return_value="/tmp/dir"):
        
        # If find_hook returns [] in the first check, it should return repo_dir immediately
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
```


