####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Teardown
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        result = find_hook('non_existent_hook', hooks_dir)
        assert result is None

    # Test with non-existent hooks directory
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

    # Test with invalid hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create invalid hook files
        invalid_files = [
            'pre_gen_project.txt',  # wrong extension
            'invalid_hook.py',      # not in _HOOKS
            'pre_gen_project.py~',  # backup file
        ]
        for filename in invalid_files:
            with open(os.path.join(hooks_dir, filename), 'w') as f:
                f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #3
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            assert os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')
            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')
            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            assert os.path.exists(project_dir)

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('{{ undefined_variable }}')
            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            assert os.path.exists(project_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hook_script = Path(repo_dir) / 'hooks' / 'post_gen_project.py'
            hook_script.parent.mkdir()
            hook_script.write_text('print("Hook executed")')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = Path(repo_dir) / 'hooks' / 'post_gen_project.py'
            hook_script.parent.mkdir()
            hook_script.write_text('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
                )
            assert not Path(project_dir).exists()

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = Path(repo_dir) / 'hooks' / 'post_gen_project.py'
            hook_script.parent.mkdir()
            hook_script.write_text('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, False
                )
            assert Path(project_dir).exists()

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a hook script with undefined variable
            hook_script = Path(repo_dir) / 'hooks' / 'post_gen_project.py'
            hook_script.parent.mkdir()
            hook_script.write_text('{{ undefined_variable }}')

            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
                )
            assert not Path(project_dir).exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, True
            )


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.sh', '/tmp/repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook():
    # Test that run_hook executes the correct hook script
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock project directory
        project_dir = Path(temp_dir) / 'project'
        project_dir.mkdir()

        # Create a mock hook script
        hook_dir = Path(temp_dir) / 'hooks'
        hook_dir.mkdir()
        hook_script = hook_dir / 'post_gen_project.py'
        hook_script.write_text('print("Hook executed")')

        # Create a mock context
        context = {'project_name': 'test_project'}

        # Change to the temp directory to simulate being in the project template
        with work_in(temp_dir):
            # Test that the hook is found and executed
            run_hook('post_gen_project', project_dir, context)

            # Verify that the hook script was executed
            # (In a real test, you might check for side effects or output)
            assert hook_script.exists()

        # Test that no hook is found when it doesn't exist
        with work_in(temp_dir):
            run_hook('nonexistent_hook', project_dir, context)
            # Should not raise an exception, just log and return

        # Test that invalid hooks are not executed
        invalid_hook = hook_dir / 'invalid_hook.txt'
        invalid_hook.write_text('print("This should not execute")')
        with work_in(temp_dir):
            run_hook('invalid_hook', project_dir, context)
            # Should not raise an exception

        # Test that backup files are not executed
        backup_hook = hook_dir / 'post_gen_project.py~'
        backup_hook.write_text('print("This should not execute")')
        with work_in(temp_dir):
            run_hook('post_gen_project', project_dir, context)
            # Should not raise an exception


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook():
    # Test valid hook file
    assert valid_hook('pre_gen_project.py', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.sh', 'post_gen_project') is True

    # Test invalid hook file (wrong name)
    assert valid_hook('pre_gen_project.py', 'post_gen_project') is False

    # Test unsupported hook file
    assert valid_hook('invalid_hook.py', 'invalid_hook') is False

    # Test backup file
    assert valid_hook('pre_gen_project.py~', 'pre_gen_project') is False

    # Test file with wrong extension
    assert valid_hook('pre_gen_project.txt', 'pre_gen_project') is False


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / 'pre_gen_project.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])

    # Mock run_script_with_context
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    run_script_with_context.assert_called_once_with(script_path, project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'non_existent_hook'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script_with_context():
    # Test with a simple script
    script_path = Path('test_script.sh')
    script_path.write_text('echo "Hello, {{ name }}!"')
    context = {'name': 'World'}

    with tempfile.TemporaryDirectory() as tmpdir:
        run_script_with_context(script_path, tmpdir, context)

    # Test with a Python script
    script_path = Path('test_script.py')
    script_path.write_text('print("Hello, {{ name }}!")')
    context = {'name': 'World'}

    with tempfile.TemporaryDirectory() as tmpdir:
        run_script_with_context(script_path, tmpdir, context)

    # Test with a script that fails
    script_path = Path('test_script_fail.sh')
    script_path.write_text('exit 1')
    context = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FailedHookException):
            run_script_with_context(script_path, tmpdir, context)

    # Test with a script that has an undefined variable
    script_path = Path('test_script_undefined.sh')
    script_path.write_text('echo "Hello, {{ undefined_var }}!"')
    context = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(UndefinedError):
            run_script_with_context(script_path, tmpdir, context)


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    context = {"key": "value"}
    delete_project_on_failure = True

    # Create necessary directories
    repo_dir.mkdir()
    project_dir.mkdir()

    # Mock dependencies
    mocker.patch("cookiecutter.hooks.work_in")
    mocker.patch("cookiecutter.hooks.run_hook")
    mocker.patch("cookiecutter.hooks.rmtree")
    mocker.patch("cookiecutter.hooks.logger")

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, delete_project_on_failure)
    cookiecutter.hooks.run_hook.assert_called_once_with("pre_gen_project", project_dir, context)

    # Test FailedHookException
    cookiecutter.hooks.run_hook.side_effect = FailedHookException("Test error")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, delete_project_on_failure)
    cookiecutter.hooks.rmtree.assert_called_once_with(project_dir)

    # Test UndefinedError
    cookiecutter.hooks.run_hook.side_effect = UndefinedError("Test error")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, delete_project_on_failure)
    assert cookiecutter.hooks.rmtree.call_count == 2

    # Test delete_project_on_failure=False
    cookiecutter.hooks.rmtree.reset_mock()
    cookiecutter.hooks.run_hook.side_effect = FailedHookException("Test error")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, False)
    cookiecutter.hooks.rmtree.assert_not_called()


# LLM-generated content at query #11
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook_path, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Valid hook")')

        # Test finding the valid hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_path]

    # Test no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', os.path.join(tmpdir, 'nonexistent'))
        assert result is None

    # Test no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a non-matching hook file
        invalid_hook_path = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(invalid_hook_path, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Invalid hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook_path = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook_path, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Backup hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        hook1_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook1_path, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hook 1")')

        hook2_path = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook2_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hook 2"')

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook1_path in result
        assert hook2_path in result


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    hook_name = "post_gen_project"
    context = {"project_name": "test_project"}
    delete_project_on_failure = True

    # Create a dummy hook script
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / f"{hook_name}.py"
    hook_script.write_text("print('Hook executed')")

    # Mock the work_in context manager
    mocker.patch("cookiecutter.hooks.work_in", lambda x: x)

    # Mock the run_hook function to avoid actual execution
    mocker.patch("cookiecutter.hooks.run_hook")

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert run_hook.called

    # Reset mocks
    run_hook.reset_mock()

    # Test failure with delete_project_on_failure=True
    run_hook.side_effect = FailedHookException("Hook failed")
    mocker.patch("cookiecutter.hooks.rmtree")
    mocker.patch("cookiecutter.hooks.logger.exception")

    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    assert rmtree.called
    assert logger.exception.called

    # Reset mocks
    run_hook.side_effect = None
    rmtree.reset_mock()
    logger.exception.reset_mock()

    # Test failure with delete_project_on_failure=False
    run_hook.side_effect = FailedHookException("Hook failed")
    delete_project_on_failure = False

    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    assert not rmtree.called
    assert logger.exception.called


# LLM-generated content at query #13
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Teardown
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook():
    # Test with non-existent hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)

        # Test with empty hooks directory
        assert find_hook('pre_gen_project', hooks_dir) is None

        # Test with valid hook file
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(valid_hook_file, 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

        # Test with invalid hook file (wrong name)
        invalid_hook_file = os.path.join(hooks_dir, 'invalid_hook.sh')
        with open(invalid_hook_file, 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

        # Test with backup file (should be ignored)
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.sh~')
        with open(backup_hook_file, 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

        # Test with multiple valid hook files
        another_valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(another_valid_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert set(result) == {valid_hook_file, another_valid_hook_file}


# LLM-generated content at query #15
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(valid_hook).touch()

        # Create an invalid hook file (wrong name)
        invalid_hook = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(invalid_hook).touch()

        # Create a backup file (should be ignored)
        backup_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(backup_hook).touch()

        # Test finding the valid hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook]

        # Test finding a non-existent hook
        result = find_hook('non_existent_hook', hooks_dir)
        assert result is None

        # Test with non-existent hooks directory
        result = find_hook('pre_gen_project', 'non_existent_dir')
        assert result is None


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            run_hook('pre_gen_project', '/fake/project_dir', {'key': 'value'})
            mock_find_hook.assert_called_once_with('pre_gen_project')
            mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook_script.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook('pre_gen_project', '/fake/project_dir', {'key': 'value'})
                mock_find_hook.assert_called_once_with('pre_gen_project')
                mock_run_script.assert_called_once_with('/fake/hook_script.py', '/fake/project_dir', {'key': 'value'})
                mock_logger.debug.assert_called_once_with('Running hook %s', 'pre_gen_project')

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook_script1.py', '/fake/hook_script2.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook('post_gen_project', '/fake/project_dir', {'key': 'value'})
                mock_find_hook.assert_called_once_with('post_gen_project')
                assert mock_run_script.call_count == 2
                mock_run_script.assert_any_call('/fake/hook_script1.py', '/fake/project_dir', {'key': 'value'})
                mock_run_script.assert_any_call('/fake/hook_script2.py', '/fake/project_dir', {'key': 'value'})
                mock_logger.debug.assert_called_once_with('Running hook %s', 'post_gen_project')


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hook_script.py'
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])

    # Mock run_script_with_context
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    run_script_with_context.assert_called_once_with(script_path, project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'non_existent_hook'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #18
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    context = {'name': 'World'}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(script_content)
        temp_script_path = temp_script.name

    # Execute
    run_script_with_context(temp_script_path, '.', context)

    # Cleanup
    os.unlink(temp_script_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Valid hook")')

        # Test finding the valid hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

    # Test no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', os.path.join(tmpdir, 'nonexistent'))
        assert result is None

    # Test no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a non-matching hook file
        non_matching_hook = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(non_matching_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Invalid hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Backup hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook1, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hook 1")')
        with open(hook2, 'w') as f:
            f.write('#!/bin/sh\necho "Hook 2"')

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook1 in result
        assert hook2 in result


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text('print("Hello, {{ name }}!")', encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Teardown
    script_path.unlink()
    os.rmdir(cwd)


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (backup file)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('fake_repo_dir')
        assert result == 'fake_repo_dir'

    # Test when pre_prompt hook is found and runs successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['fake_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('fake_repo_dir')
        assert result == 'temp_dir'
        mock_run_script.assert_called_once_with('fake_script.py', 'temp_dir')

    # Test when pre_prompt hook is found but fails
    with patch('cookiecutter.hooks.find_hook', return_value=['fake_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('fake_repo_dir')
        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'


# LLM-generated content at query #23
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
        mock_find_hook.assert_called_once_with('pre_prompt')

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)

        assert result == Path('/fake/tmp/repo')
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_run_script.side_effect = FailedHookException('Script failed')
        repo_dir = Path('/fake/repo')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook(repo_dir)

        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))


# LLM-generated content at query #24
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    with pytest.raises(FailedHookException):
        find_hook('pre_gen_project', 'nonexistent_dir')

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        assert find_hook('pre_gen_project', temp_dir) is None

    # Test case 3: Hook directory exists with matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        result = find_hook('pre_gen_project', hook_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with multiple matching hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        with open(hook_file2, 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        result = find_hook('pre_gen_project', hook_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result

    # Test case 5: Hook directory exists with invalid hook (backup file)
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        result = find_hook('pre_gen_project', hook_dir)
        assert result is None


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_with_context():
    # Test that the script is rendered with the context and executed
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text('print("Hello, {{ name }}!")', encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    run_script_with_context(script_path, cwd, context)

    # Clean up
    script_path.unlink()
    rmtree(cwd)

    # Test that a non-Python script is also rendered and executed
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text('echo "Hello, {{ name }}!"', encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    run_script_with_context(script_path, cwd, context)

    # Clean up
    script_path.unlink()
    rmtree(cwd)

    # Test that a FailedHookException is raised if the script fails
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text('import sys; sys.exit(1)', encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {}

    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, cwd, context)

    # Clean up
    script_path.unlink()
    rmtree(cwd)


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['test_script.py']
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = Mock()
        mock_work_in.return_value.__exit__ = Mock()

        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['test_script.py']
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = Mock()
        mock_work_in.return_value.__exit__ = Mock()
        mock_run_script.side_effect = FailedHookException('test error')

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #27
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 3: Hook directory exists with matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with multiple matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result

    # Test case 5: Hook directory exists with backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 6: Hook directory exists with invalid hook name
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None


# LLM-generated content at query #28
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'

    # Test when pre_prompt hook is found and runs successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo_dir/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/tmp/fake_repo'
        mock_run_script.assert_called_once_with('/fake/repo_dir/hooks/pre_prompt.py', '/tmp/fake_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo_dir/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Hook failed')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('/fake/repo_dir')
        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'


# LLM-generated content at query #29
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a valid hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed successfully")')

            context = {'project_name': 'test_project'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test_project'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test_project'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test with UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a hook script that will raise UndefinedError
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('{{ undefined_variable }}')

            context = {'project_name': 'test_project'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test with non-existent hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test_project'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='non_existent_hook',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
            assert os.path.exists(project_dir)


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(valid_hook_file).touch()

        # Test finding the valid hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file
        invalid_hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(invalid_hook_file).touch()

        # Test finding no hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test with non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_hooks_dir = os.path.join(tmpdir, 'non_existent_hooks_dir')

        # Test with non-existent hooks directory
        result = find_hook('pre_gen_project', non_existent_hooks_dir)
        assert result is None

    # Test finding multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        valid_hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        valid_hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(valid_hook_file1).touch()
        Path(valid_hook_file2).touch()

        # Test finding multiple valid hooks
        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert valid_hook_file1 in result
        assert valid_hook_file2 in result

    # Test with backup file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(backup_hook_file).touch()

        # Test with backup file
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #31
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'project_name': 'test_project'}
    hook_name = 'pre_gen_project'

    # Create a simple hook script
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, f'{hook_name}.py')
    with open(hook_script, 'w') as f:
        f.write('#!/usr/bin/env python\nprint("Hook executed")')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert os.path.exists(project_dir)

    # Test failure with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('#!/usr/bin/env python\nimport sys\nsys.exit(1)')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Test failure with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    context = {"test": "value"}
    delete_project_on_failure = True

    # Create repo and project directories
    repo_dir.mkdir()
    project_dir.mkdir()

    # Create a hook directory and script
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/sh\necho 'test'")

    # Mock work_in to change directory to repo_dir
    mocker.patch('cookiecutter.hooks.work_in', lambda x: x)

    # Mock rmtree to avoid actual deletion
    mocker.patch('cookiecutter.hooks.rmtree')

    # Test successful hook execution
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)

    # Test FailedHookException
    hook_script.write_text("#!/bin/sh\nexit 1")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)

    # Test UndefinedError
    hook_script.write_text("#!/bin/sh\necho '{{ undefined_variable }}'")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test script with OSError (missing shebang)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('echo "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #34
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (backup file)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'wrong_name.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #35
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.sh']):
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=Path('/fake/tmp/repo')):
                repo_dir = Path('/fake/repo')
                result = run_pre_prompt_hook(repo_dir)
                assert result == Path('/fake/tmp/repo')
                mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.sh', '/fake/tmp/repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.sh']):
        with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Test error')):
            with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=Path('/fake/tmp/repo')):
                repo_dir = Path('/fake/repo')
                with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
                    run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    with pytest.raises(FailedHookException):
        find_hook('pre_gen_project', 'nonexistent_dir')

    # Test case 2: No matching hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'invalid_hook.py'), 'w') as f:
            f.write('# invalid hook')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 3: Valid hook file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('# valid hook')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test case 4: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('# valid hook 1')
        with open(hook_file2, 'w') as f:
            f.write('# valid hook 2')
        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result

    # Test case 5: Backup file should be ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('# backup hook')
        assert find_hook('pre_gen_project', hooks_dir) is None


# LLM-generated content at query #37
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/path/to/script.py'])

    # Mock run_script_with_context
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    mock_run_script.assert_called_once_with('/path/to/script.py', project_dir, context)

def test_run_hook_no_script(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None (no script found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #38
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook exists
    with tempfile.TemporaryDirectory() as temp_dir:
        assert run_pre_prompt_hook(temp_dir) == temp_dir

    # Test when pre_prompt hook exists and succeeds
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello")')
        result = run_pre_prompt_hook(temp_dir)
        assert result != temp_dir
        assert os.path.exists(result)

    # Test when pre_prompt hook exists and fails
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_file, 'w') as f:
            f.write('import sys; sys.exit(1)')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook(temp_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        utils.make_executable(script_path)
        run_script(script_path, cwd=tmpdir)

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'failing_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        utils.make_executable(script_path)
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_python.py')
        with open(script_path, 'w') as f:
            f.write('print("Python script executed")\n')
        run_script(script_path, cwd=tmpdir)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'no_shebang.sh')
        with open(script_path, 'w') as f:
            f.write('echo "No shebang"\n')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test non-existent script
    with pytest.raises(FailedHookException) as excinfo:
        run_script('/nonexistent/script.sh')
    assert 'Hook script failed (error:' in str(excinfo.value)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_script_with_context(mocker, tmp_path):
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tmp_path
    context = {'name': 'World'}

    # Mock subprocess.Popen and its wait method
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.MagicMock()
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc

    # Mock utils.make_executable
    mocker.patch('cookiecutter.utils.make_executable')

    # Mock create_env_with_context
    mock_env = mocker.MagicMock()
    mock_template = mocker.MagicMock()
    mock_template.render.return_value = "print('Hello, World!')"
    mock_env.from_string.return_value = mock_template
    mocker.patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env)

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Assert
    mock_popen.assert_called_once_with(
        [sys.executable, str(script_path)],
        shell=False,
        cwd=cwd
    )
    mock_proc.wait.assert_called_once()


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failed execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test execution of a non-Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test execution of a non-existent script
    with pytest.raises(FailedHookException):
        run_script('/path/to/nonexistent/script.py')

    # Test execution of an empty script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/path/to/script.py'])

    # Mock run_script_with_context
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    mock_run_script.assert_called_once_with('/path/to/script.py', project_dir, context)

def test_run_hook_no_script(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None (no script found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #43
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello"\n')
        utils.make_executable(script_path)
        run_script(script_path, tmpdir)

    # Test failed script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        utils.make_executable(script_path)
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello")\n')
        run_script(script_path, tmpdir)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello"\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test successful execution of a shell script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello, World!"\n')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure of a script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure of a script with no shebang
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('echo "Hello, World!"\n')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook', return_value=['/path/to/script.py'])
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    mock_run_script_with_context.assert_called_once_with('/path/to/script.py', project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'non_existent_hook'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    mock_run_script_with_context.assert_not_called()
    assert f'No {hook_name} hook found' in caplog.text

def test_run_hook_multiple_scripts(mocker, tmp_path):
    # Setup
    hook_name = 'post_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return multiple scripts
    scripts = ['/path/to/script1.py', '/path/to/script2.sh']
    mocker.patch('cookiecutter.hooks.find_hook', return_value=scripts)
    mock_run_script_with_context = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    mock_run_script_with_context.assert_has_calls([
        mocker.call(scripts[0], project_dir, context),
        mocker.call(scripts[1], project_dir, context)
    ], any_order=False)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'test_repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Test error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()

        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result == [hook_file]
        finally:
            os.chdir(original_cwd)

    # Test no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)

    # Test invalid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file
        invalid_hook = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(invalid_hook).touch()

        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(backup_hook).touch()

        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_path = Path('test_script.py')
    script_path.write_text('print("Hello, {{ name }}!")', encoding='utf-8')
    cwd = Path('.')
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Teardown
    script_path.unlink()


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            # Run the hook
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

    # Test hook failure with project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect failure
            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )

            # Verify project directory was deleted
            assert not os.path.exists(project_dir)

    # Test hook failure without project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect failure
            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )

            # Verify project directory still exists
            assert os.path.exists(project_dir)

    # Test with no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Run the hook without any hooks present
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

            # Verify project directory still exists
            assert os.path.exists(project_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/repo'
        mock_work_in.return_value.__enter__ = mock_work_in

        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/repo'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Script failed')

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test successful execution of a shell script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello, World!"\n')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure with ENOEXEC error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('This is not executable')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\necho "Hello, World!"\n')
        run_script(str(script_path), tmpdir)

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'failing_script.sh'
        script_path.write_text('#!/bin/sh\nexit 1\n')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, Python!")\n')
        run_script(str(script_path), tmpdir)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'noshebang_script.sh'
        script_path.write_text('echo "No shebang"\n')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), tmpdir)

    # Test script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'nonexistent_script.sh'
        with pytest.raises(FailedHookException):
            run_script(str(script_path), tmpdir)


# LLM-generated content at query #8
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('test error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\necho "Hello, World!"\n')
        run_script(str(script_path), cwd=tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, World!")\n')
        run_script(str(script_path), cwd=tmpdir)

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\nexit 1\n')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), cwd=tmpdir)

    # Test script with OSError (missing shebang)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('echo "Hello, World!"\n')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), cwd=tmpdir)


# LLM-generated content at query #10
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Teardown
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test when no hook is found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_mock = mocker.patch('cookiecutter.hooks.logger')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    logger_mock.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test when hook is found and executed successfully
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert hook_script.exists()

    # Test when hook execution fails
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('exit(1)')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    mocker.patch('cookiecutter.hooks.run_script_with_context', side_effect=FailedHookException('Hook failed'))
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})


# LLM-generated content at query #12
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Cleanup
    script_path.unlink()
    rmtree(cwd)


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test when no hook is found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_mock = mocker.patch('cookiecutter.hooks.logger')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    logger_mock.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test when hook is found and executed
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert hook_script.exists()


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test when no hook is found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_debug = mocker.patch('cookiecutter.hooks.logger.debug')
    run_hook('pre_gen_project', tmp_path, {})
    logger_debug.assert_called_with('No %s hook found', 'pre_gen_project')

    # Test when hook is found and executed
    script_path = tmp_path / 'hook.py'
    script_path.write_text('print("test")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'test': 'value'})
    run_script_with_context_mock.assert_called_once_with(str(script_path), tmp_path, {'test': 'value'})


# LLM-generated content at query #15
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(valid_hook_file).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

    # Test no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', os.path.join(tmpdir, 'nonexistent'))
        assert result is None

    # Test no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a non-matching hook file
        non_matching_hook = os.path.join(hooks_dir, 'other_script.py')
        Path(non_matching_hook).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(backup_hook).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test unsupported hook is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an unsupported hook file
        unsupported_hook = os.path.join(hooks_dir, 'unsupported_hook.py')
        Path(unsupported_hook).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'project_name': 'test_project'}
    hook_name = 'pre_gen_project'

    # Create a simple hook script
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, f'{hook_name}.py')
    with open(hook_script, 'w') as f:
        f.write('print("Hook executed")')

    # Test successful execution
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except Exception as e:
        pytest.fail(f"Hook execution failed: {e}")
    finally:
        rmtree(repo_dir)
        rmtree(project_dir)

    # Test failure with delete_project_on_failure=True
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, f'{hook_name}.py')
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')

    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)
    rmtree(repo_dir)

    # Test failure with delete_project_on_failure=False
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, f'{hook_name}.py')
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')

    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)
    rmtree(repo_dir)
    rmtree(project_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    except FailedHookException:
        pytest.fail("Script execution failed unexpectedly")
    finally:
        os.unlink(script_path)

    # Test script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
        pytest.fail("Expected FailedHookException for non-zero exit status")
    except FailedHookException as e:
        assert "exit status: 1" in str(e)
    finally:
        os.unlink(script_path)

    # Test script with OSError (missing shebang)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('echo "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
        pytest.fail("Expected FailedHookException for missing shebang")
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

    # Test script with other OSError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/bash\necho "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    except FailedHookException:
        pytest.fail("Script execution failed unexpectedly")
    finally:
        os.unlink(script_path)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo_dir/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__.return_value = None

        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/tmp/fake_repo'
        mock_run_script.assert_called_once_with('/fake/repo_dir/hooks/pre_prompt.py', '/tmp/fake_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo_dir/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #19
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure of a script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test failure of a script with OSError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            run_hook('pre_gen_project', '/fake/project/dir', {'key': 'value'})
            mock_logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook/script.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('pre_gen_project', '/fake/project/dir', {'key': 'value'})
            mock_run_script.assert_called_once_with('/fake/hook/script.py', '/fake/project/dir', {'key': 'value'})

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook/script1.py', '/fake/hook/script2.sh']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('post_gen_project', '/fake/project/dir', {'key': 'value'})
            assert mock_run_script.call_count == 2
            mock_run_script.assert_any_call('/fake/hook/script1.py', '/fake/project/dir', {'key': 'value'})
            mock_run_script.assert_any_call('/fake/hook/script2.sh', '/fake/project/dir', {'key': 'value'})


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_with_context():
    # Test basic script execution with context
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    context = {'name': 'World'}

    run_script_with_context(script_path, '.', context)

    # Test non-Python script execution
    script_content = "echo 'Hello, {{ name }}!'"
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text(script_content, encoding='utf-8')

    run_script_with_context(script_path, '.', context)

    # Test script with missing context variable
    script_content = "print('Hello, {{ missing_var }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    context = {'name': 'World'}

    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, '.', context)

    # Test script that fails during execution
    script_content = "import sys; sys.exit(1)"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    context = {'name': 'World'}

    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, '.', context)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    context = {'name': 'World'}
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')

    # Execute
    run_script_with_context(script_path, '.', context)

    # Teardown
    script_path.unlink()


# LLM-generated content at query #23
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            result = run_pre_prompt_hook('/fake/repo_dir')
            assert result == '/fake/repo_dir'
            mock_work_in.assert_called_once_with('/fake/repo_dir')

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']):
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo_dir'):
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    result = run_pre_prompt_hook('/fake/repo_dir')
                    assert result == '/tmp/repo_dir'
                    mock_work_in.assert_called_with('/tmp/repo_dir')
                    mock_run_script.assert_called_once_with('/fake/script.py', '/tmp/repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']):
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo_dir'):
                with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
                    with pytest.raises(FailedHookException) as excinfo:
                        run_pre_prompt_hook('/fake/repo_dir')
                    assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #24
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        os.chmod(script_path, 0o755)
        run_script(script_path, tmpdir)

    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")\n')
        run_script(script_path, tmpdir)

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        os.chmod(script_path, 0o755)
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test script with OSError (missing shebang)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"\n')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test script with OSError (other error)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        os.chmod(script_path, 0o755)
        with mock.patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError(errno.EACCES, 'Permission denied')
            with pytest.raises(FailedHookException) as excinfo:
                run_script(script_path, tmpdir)
            assert 'Hook script failed (error: [Errno 13] Permission denied)' in str(excinfo.value)


# LLM-generated content at query #25
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
        mock_find_hook.assert_called_once_with('pre_prompt')

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        repo_dir = Path('/fake/repo')

        result = run_pre_prompt_hook(repo_dir)

        assert result == Path('/fake/tmp/repo')
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_run_script.side_effect = FailedHookException('Test failure')
        repo_dir = Path('/fake/repo')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook(repo_dir)

        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))


# LLM-generated content at query #26
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test when no hook is found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_debug = mocker.patch('cookiecutter.hooks.logger.debug')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    logger_debug.assert_called_with('No %s hook found', 'pre_gen_project')

    # Test when hook is found and executed
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    run_script_with_context_mock.assert_called_once_with(str(hook_script), tmp_path, {'project_name': 'test'})


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        run_hook('pre_gen_project', '/fake/project_dir', {'fake': 'context'})
        mock_find_hook.assert_called_once_with('pre_gen_project')

    # Test case 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/fake/hook_script.sh']
        run_hook('post_gen_project', '/fake/project_dir', {'fake': 'context'})
        mock_find_hook.assert_called_once_with('post_gen_project')
        mock_run_script.assert_called_once_with('/fake/hook_script.sh', '/fake/project_dir', {'fake': 'context'})

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/fake/hook1.sh', '/fake/hook2.py']
        run_hook('pre_gen_project', '/fake/project_dir', {'fake': 'context'})
        mock_find_hook.assert_called_once_with('pre_gen_project')
        assert mock_run_script.call_count == 2
        mock_run_script.assert_any_call('/fake/hook1.sh', '/fake/project_dir', {'fake': 'context'})
        mock_run_script.assert_any_call('/fake/hook2.py', '/fake/project_dir', {'fake': 'context'})


# LLM-generated content at query #28
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()

    hook_name = "post_gen_project"
    context = {"project_name": "test_project"}
    delete_project_on_failure = True

    # Create a valid hook script
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / f"{hook_name}.py"
    hook_script.write_text("print('Hook executed')")

    # Mock work_in to avoid changing directories
    mocker.patch("cookiecutter.hooks.work_in", lambda x: mocker.MagicMock())

    # Mock run_hook to avoid actual execution
    mocker.patch("cookiecutter.hooks.run_hook")

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert run_hook.called

    # Test failure with delete_project_on_failure=True
    run_hook.side_effect = FailedHookException("Hook failed")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert not project_dir.exists()

    # Test failure with delete_project_on_failure=False
    project_dir.mkdir()
    delete_project_on_failure = False
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert project_dir.exists()

    # Test UndefinedError
    run_hook.side_effect = UndefinedError("Undefined variable")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert not project_dir.exists()


# LLM-generated content at query #29
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    hook_name = "pre_gen_project"
    context = {"project_name": "test"}

    # Create a valid hook script
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / f"{hook_name}.py"
    hook_script.write_text("print('Hook executed')")

    # Mock the work_in context manager
    work_in_mock = mocker.patch('cookiecutter.hooks.work_in')
    work_in_mock.return_value.__enter__.return_value = None

    # Mock the run_hook function
    run_hook_mock = mocker.patch('cookiecutter.hooks.run_hook')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    run_hook_mock.assert_called_once_with(hook_name, project_dir, context)

    # Test failed execution with delete_project_on_failure=True
    run_hook_mock.side_effect = FailedHookException("Hook failed")
    rmtree_mock = mocker.patch('cookiecutter.hooks.rmtree')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    rmtree_mock.assert_called_once_with(project_dir)

    # Test failed execution with delete_project_on_failure=False
    rmtree_mock.reset_mock()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    rmtree_mock.assert_not_called()

    # Test UndefinedError exception
    run_hook_mock.side_effect = UndefinedError("Undefined variable")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)


# LLM-generated content at query #30
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')
            context = {'test': 'value'}
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')
            context = {'test': 'value'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')
            context = {'test': 'value'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'test': 'value'}
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)


