####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    with pytest.raises(AssertionError):
        find_hook('pre_gen_project', 'nonexistent_dir')

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'other_script.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 3: Hook directory exists with matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0].endswith('pre_gen_project.py')

    # Test case 4: Hook directory exists with multiple matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
            f.write('#!/bin/sh\necho "Hello"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert all(script.endswith(('pre_gen_project.py', 'pre_gen_project.sh')) for script in result)

    # Test case 5: Hook directory exists with backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0].endswith('pre_gen_project.py')


# LLM-generated content at query #2
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

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('test error')):
        with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
            run_pre_prompt_hook('test_repo_dir')


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')
        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')
        run_script(script_path, cwd=tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test failure of a script with no shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test failure of a script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(0)')
        os.chmod(script_path, 0o000)  # Make the script non-executable
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed (error:' in str(excinfo.value)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('/fake/repo')


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test case 1: No hook found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mocker.patch('cookiecutter.hooks.logger')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    cookiecutter.hooks.logger.debug.assert_called_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed successfully
    script_path = tmp_path / 'hooks' / 'pre_gen_project.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    cookiecutter.hooks.run_script_with_context.assert_called_once_with(
        str(script_path), tmp_path, {'project_name': 'test'}
    )

    # Test case 3: Multiple hooks found and executed
    script_path1 = tmp_path / 'hooks' / 'pre_gen_project.py'
    script_path2 = tmp_path / 'hooks' / 'pre_gen_project.sh'
    script_path1.parent.mkdir()
    script_path1.write_text('print("Hook 1 executed")')
    script_path2.write_text('echo "Hook 2 executed"')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path1), str(script_path2)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert cookiecutter.hooks.run_script_with_context.call_count == 2


# LLM-generated content at query #6
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

    # Test failure of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test non-existent script
    with pytest.raises(FailedHookException):
        run_script('/non/existent/script.py')

    # Test empty script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('')
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
def test_valid_hook():
    # Test valid hook
    assert valid_hook('pre_gen_project.py', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.sh', 'post_gen_project') is True

    # Test invalid hook name
    assert valid_hook('invalid_hook.py', 'pre_gen_project') is False

    # Test unsupported hook
    assert valid_hook('invalid_hook.py', 'invalid_hook') is False

    # Test backup file
    assert valid_hook('pre_gen_project.py~', 'pre_gen_project') is False

    # Test different extensions
    assert valid_hook('pre_prompt.sh', 'pre_prompt') is True
    assert valid_hook('pre_prompt', 'pre_prompt') is True


# LLM-generated content at query #8
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
        temp.write(b'print("Hello, World!")')
        temp_path = temp.name

    try:
        run_script(temp_path)
    finally:
        os.unlink(temp_path)

    # Test failure of a script
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
        temp.write(b'import sys; sys.exit(1)')
        temp_path = temp.name

    try:
        with pytest.raises(FailedHookException) as excinfo:
            run_script(temp_path)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)
    finally:
        os.unlink(temp_path)

    # Test non-existent script
    with pytest.raises(FailedHookException) as excinfo:
        run_script('/non/existent/script.py')
    assert 'Hook script failed (error:' in str(excinfo.value)

    # Test script without shebang
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as temp:
        temp.write(b'print("Hello, World!")')
        temp_path = temp.name

    try:
        with pytest.raises(FailedHookException) as excinfo:
            run_script(temp_path)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook():
    # Test with non-existent hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test with empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with valid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test with invalid hook file (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with multiple valid hook files
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


# LLM-generated content at query #10
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')

        run_script(script_path, tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')

        run_script(script_path, tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test failure of a script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test failure of a script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(0)')

        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError(errno.ENOEXEC, 'Test error')
            with pytest.raises(FailedHookException) as excinfo:
                run_script(script_path, tmpdir)
            assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError(errno.EACCES, 'Test error')
            with pytest.raises(FailedHookException) as excinfo:
                run_script(script_path, tmpdir)
            assert 'Hook script failed (error: Test error)' in str(excinfo.value)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / 'repo'
    project_dir = tmp_path / 'project'
    repo_dir.mkdir()
    project_dir.mkdir()

    hook_name = 'post_gen_project'
    context = {'project_name': 'test_project'}

    # Create a dummy hook script
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_script = hooks_dir / f'{hook_name}.py'
    hook_script.write_text('print("Hook executed")')

    # Mock the work_in context manager
    work_in_mock = mocker.patch('cookiecutter.utils.work_in')
    work_in_mock.return_value.__enter__.return_value = None

    # Mock run_hook to avoid actual execution
    run_hook_mock = mocker.patch('cookiecutter.hooks.run_hook')

    # Test successful execution
    run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure=True
    )
    run_hook_mock.assert_called_once_with(hook_name, project_dir, context)

    # Test FailedHookException
    run_hook_mock.side_effect = FailedHookException('Hook failed')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure=True
        )
    assert not project_dir.exists()  # Project dir should be deleted

    # Test UndefinedError
    run_hook_mock.side_effect = UndefinedError('Undefined variable')
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure=False
        )
    assert project_dir.exists()  # Project dir should not be deleted


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.work_in') as mock_work_in:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/fake_repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/fake_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.work_in') as mock_work_in:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('/fake/repo')
        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'


# LLM-generated content at query #13
#--------------------------

```python
def test_run_script(mocker, tmp_path):
    # Test successful script execution
    script_path = tmp_path / 'test_script.sh'
    script_path.write_text('#!/bin/bash\necho "Hello, World!"\n')
    mocker.patch('subprocess.Popen')
    proc_mock = mocker.MagicMock()
    proc_mock.wait.return_value = 0
    mocker.patch('subprocess.Popen', return_value=proc_mock)
    run_script(str(script_path))
    proc_mock.wait.assert_called_once()

    # Test failed script execution
    proc_mock.wait.return_value = 1
    with pytest.raises(FailedHookException):
        run_script(str(script_path))

    # Test Python script execution
    script_path = tmp_path / 'test_script.py'
    script_path.write_text('print("Hello, World!")\n')
    mocker.patch('subprocess.Popen')
    proc_mock = mocker.MagicMock()
    proc_mock.wait.return_value = 0
    mocker.patch('subprocess.Popen', return_value=proc_mock)
    run_script(str(script_path))
    proc_mock.wait.assert_called_once()

    # Test OSError with ENOEXEC
    mocker.patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'Test error'))
    with pytest.raises(FailedHookException):
        run_script(str(script_path))

    # Test OSError with other error
    mocker.patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Test error'))
    with pytest.raises(FailedHookException):
        run_script(str(script_path))


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        with work_in(tmpdir):
            run_hook('non_existent_hook', tmpdir, {})

    # Test case 2: Valid hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hook executed")')

        with work_in(tmpdir):
            run_hook('pre_gen_project', tmpdir, {})

    # Test case 3: Hook with context
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.sh')
        with open(hook_file, 'w') as f:
            f.write('echo "Project name is {{ cookiecutter.project_name }}"')

        context = {'cookiecutter': {'project_name': 'test_project'}}
        with work_in(tmpdir):
            run_hook('post_gen_project', tmpdir, context)

    # Test case 4: Hook fails
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('import sys; sys.exit(1)')

        with work_in(tmpdir):
            with pytest.raises(FailedHookException):
                run_hook('pre_gen_project', tmpdir, {})


# LLM-generated content at query #15
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'
        mock_find_hook.assert_called_once_with('pre_prompt')

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp.return_value = '/tmp/repo'
        mock_work_in.return_value.__enter__ = mock_work_in

        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/repo'
        mock_create_tmp.assert_called_once_with('/fake/repo')
        mock_run_script.assert_called_once_with('/fake/script.py', '/tmp/repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp.return_value = '/tmp/repo'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Failed')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('/fake/repo')

        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'
        assert exc_info.value.__cause__ is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_run_script_with_context(tmp_path):
    # Create a test script with Jinja2 template
    test_script = tmp_path / 'test_script.sh'
    test_script.write_text('echo "Hello, {{ name }}!"')

    # Create a context
    context = {'name': 'World'}

    # Run the script with context
    run_script_with_context(test_script, tmp_path, context)

    # Check if the rendered script was executed
    assert (tmp_path / 'test_script.sh').exists()

    # Create a Python test script with Jinja2 template
    test_py_script = tmp_path / 'test_py_script.py'
    test_py_script.write_text('print("Hello, {{ name }}!")')

    # Run the Python script with context
    run_script_with_context(test_py_script, tmp_path, context)

    # Check if the rendered Python script was executed
    assert (tmp_path / 'test_py_script.py').exists()

    # Test with a script that should fail
    failing_script = tmp_path / 'failing_script.sh'
    failing_script.write_text('exit 1')

    with pytest.raises(FailedHookException):
        run_script_with_context(failing_script, tmp_path, context)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_pre_prompt_hook(mocker, tmp_path):
    # Test case 1: No pre_prompt hook found
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

    # Test case 2: Pre_prompt hook found and executed successfully
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_prompt.py"
    script.write_text("#!/usr/bin/env python\nprint('Hello')")
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script)])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=str(repo_dir))
    mocker.patch('cookiecutter.hooks.run_script')
    result = run_pre_prompt_hook(repo_dir)
    assert result == str(repo_dir)

    # Test case 3: Pre_prompt hook found but execution failed
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_prompt.py"
    script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script)])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=str(repo_dir))
    mocker.patch('cookiecutter.hooks.run_script', side_effect=FailedHookException("Failed"))
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #19
#--------------------------

```python
def test_run_script():
    # Test successful execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write('#!/bin/sh\necho "Hello, World!"\n')
        temp_path = temp.name

    try:
        run_script(temp_path)
    finally:
        os.unlink(temp_path)

    # Test successful execution of a Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
        temp.write('print("Hello, World!")\n')
        temp_path = temp.name

    try:
        run_script(temp_path)
    finally:
        os.unlink(temp_path)

    # Test script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write('#!/bin/sh\nexit 1\n')
        temp_path = temp.name

    try:
        with pytest.raises(FailedHookException) as excinfo:
            run_script(temp_path)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)
    finally:
        os.unlink(temp_path)

    # Test script with OSError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write('#!/bin/sh\nnonexistent_command\n')
        temp_path = temp.name

    try:
        with pytest.raises(FailedHookException) as excinfo:
            run_script(temp_path)
        assert 'Hook script failed' in str(excinfo.value)
    finally:
        os.unlink(temp_path)

    # Test script with ENOEXEC error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write('echo "Hello, World!"\n')  # No shebang
        temp_path = temp.name

    try:
        with pytest.raises(FailedHookException) as excinfo:
            run_script(temp_path)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #20
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

        # Create a backup hook file
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('print("Hook executed")')

        context = {'project_name': 'test'}

        run_hook_from_repo_dir(
            repo_dir,
            'post_gen_project',
            project_dir,
            context,
            delete_project_on_failure=True
        )

        assert project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('import sys; sys.exit(1)')

        context = {'project_name': 'test'}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

        assert not project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('import sys; sys.exit(1)')

        context = {'project_name': 'test'}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=False
            )

        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {'project_name': 'test'}

        run_hook_from_repo_dir(
            repo_dir,
            'post_gen_project',
            project_dir,
            context,
            delete_project_on_failure=True
        )

        assert project_dir.exists()


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'test_repo_dir'
        mock_find_hook.assert_called_once_with('pre_prompt')

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['pre_prompt_script'], ['pre_prompt_script']]
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_work_in.return_value.__exit__ = mock_work_in

        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'temp_repo_dir'
        mock_find_hook.assert_has_calls([call('pre_prompt'), call('pre_prompt')])
        mock_create_tmp_repo_dir.assert_called_once_with('test_repo_dir')
        mock_run_script.assert_called_once_with('pre_prompt_script', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['pre_prompt_script'], ['pre_prompt_script']]
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_work_in.return_value.__exit__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert str(excinfo.value) == 'Pre-Prompt Hook script failed'
        mock_find_hook.assert_has_calls([call('pre_prompt'), call('pre_prompt')])
        mock_create_tmp_repo_dir.assert_called_once_with('test_repo_dir')
        mock_run_script.assert_called_once_with('pre_prompt_script', 'temp_repo_dir')


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\necho "Hook executed"\n')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1\n')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1\n')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test with non-existent hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'non_existent_hook',
                project_dir,
                context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'test': 'value'}
    hook_name = 'pre_gen_project'
    hook_script = os.path.join(repo_dir, 'hooks', f'{hook_name}.py')
    os.makedirs(os.path.dirname(hook_script), exist_ok=True)
    with open(hook_script, 'w') as f:
        f.write('exit(0)')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)

    # Test failed execution with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('exit(1)')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Test failed execution with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)

    # Test with UndefinedError
    with open(hook_script, 'w') as f:
        f.write('{{ undefined_variable }}')
    project_dir = tempfile.mkdtemp()
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('test_repo')


# LLM-generated content at query #26
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test case 1: No hook found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_debug = mocker.patch('cookiecutter.hooks.logger.debug')
    run_hook('pre_gen_project', tmp_path, {})
    logger_debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('print("Hello")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    run_script_with_context_mock.assert_called_once_with(str(hook_script), tmp_path, {'project_name': 'test'})

    # Test case 3: Multiple hooks found and executed
    hook_script1 = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script2 = tmp_path / 'hooks' / 'pre_gen_project.sh'
    hook_script1.parent.mkdir()
    hook_script1.write_text('print("Hello")')
    hook_script2.write_text('echo "Hello"')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script1), str(hook_script2)])
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert run_script_with_context_mock.call_count == 2
    run_script_with_context_mock.assert_any_call(str(hook_script1), tmp_path, {'project_name': 'test'})
    run_script_with_context_mock.assert_any_call(str(hook_script2), tmp_path, {'project_name': 'test'})


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_path, 'w') as f:
                f.write('print("Hook executed")')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_path, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_path, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_path, 'w') as f:
                f.write('{{ undefined_variable }}')

            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )
            assert os.path.exists(project_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['script1.py', 'script2.sh']
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        assert mock_run_script.call_count == 2

    # Test case 3: Pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['script1.py']
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('test_repo')


# LLM-generated content at query #29
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

    # Test failed execution of a script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        os.chmod(script_path, 0o755)
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)

    # Test execution of an empty script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('')
        os.chmod(script_path, 0o755)
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)

    # Test execution of a script with no shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"\n')
        os.chmod(script_path, 0o755)
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)


# LLM-generated content at query #30
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

        mock_find_hook.side_effect = [['/fake/repo/hooks/pre_prompt.py'], ['/fake/repo/hooks/pre_prompt.py']]
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_work_in.return_value.__enter__ = mock_work_in

        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)

        assert result == Path('/fake/tmp/repo')
        mock_find_hook.assert_has_calls([call('pre_prompt'), call('pre_prompt')])
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['/fake/repo/hooks/pre_prompt.py'], ['/fake/repo/hooks/pre_prompt.py']]
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Script failed')

        repo_dir = Path('/fake/repo')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook(repo_dir)

        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'
        mock_find_hook.assert_has_calls([call('pre_prompt'), call('pre_prompt')])
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(Path('/fake/tmp/repo')))


# LLM-generated content at query #31
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')

    # Test case 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/tmp/hook_script.py']
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')
        mock_run_script.assert_called_once_with('/tmp/hook_script.py', '/tmp/project', {'name': 'test'})

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/tmp/hook_script1.py', '/tmp/hook_script2.py']
        run_hook('post_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('post_gen_project')
        assert mock_run_script.call_count == 2
        mock_run_script.assert_any_call('/tmp/hook_script1.py', '/tmp/project', {'name': 'test'})
        mock_run_script.assert_any_call('/tmp/hook_script2.py', '/tmp/project', {'name': 'test'})


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a valid hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'pre_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\necho "Hook executed"')

            context = {'project_name': 'test'}

            # Should not raise any exception
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='pre_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'pre_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}

            # Should raise FailedHookException and delete project_dir
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='pre_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )

            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'pre_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}

            # Should raise FailedHookException but not delete project_dir
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='pre_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )

            assert os.path.exists(project_dir)

    # Test with no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}

            # Should not raise any exception
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='pre_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #33
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

    # Test case 5: Hook directory exists with invalid hook (backup file)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 6: Hook directory exists with invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None


# LLM-generated content at query #34
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a valid hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\necho "Hook executed"')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True,
            )

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True,
                )
            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project')
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False,
                )
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True,
            )


# LLM-generated content at query #35
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            mock_find_hook.return_value = None
            repo_dir = Path('/tmp/test_repo')
            result = run_pre_prompt_hook(repo_dir)
            assert result == repo_dir
            mock_find_hook.assert_called_once_with('pre_prompt')

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            with patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir:
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_find_hook.return_value = ['/tmp/test_repo/hooks/pre_prompt.py']
                    mock_create_tmp_repo_dir.return_value = Path('/tmp/test_repo_temp')
                    repo_dir = Path('/tmp/test_repo')
                    result = run_pre_prompt_hook(repo_dir)
                    assert result == Path('/tmp/test_repo_temp')
                    mock_find_hook.assert_called_with('pre_prompt')
                    mock_run_script.assert_called_once_with('/tmp/test_repo_temp/hooks/pre_prompt.py', str(Path('/tmp/test_repo_temp')))

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            with patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir:
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_find_hook.return_value = ['/tmp/test_repo/hooks/pre_prompt.py']
                    mock_create_tmp_repo_dir.return_value = Path('/tmp/test_repo_temp')
                    mock_run_script.side_effect = FailedHookException('Test error')
                    repo_dir = Path('/tmp/test_repo')
                    with pytest.raises(FailedHookException) as excinfo:
                        run_pre_prompt_hook(repo_dir)
                    assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #36
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\necho "test"\n')

        context = {'test': 'value'}
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\nexit 1\n')

        context = {'test': 'value'}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert not project_dir.exists()

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\nexit 1\n')

        context = {'test': 'value'}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {'test': 'value'}
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)


# LLM-generated content at query #37
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])

    # Mock run_script_with_context
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert run_script_with_context.called
    run_script_with_context.assert_called_once_with(str(script_path), project_dir, context)

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
    assert not run_script_with_context.called
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #38
#--------------------------

```python
def test_find_hook():
    # Test with non-existent hooks directory
    assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Test with empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test with valid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test with invalid hook file (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with multiple valid hook files
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


# LLM-generated content at query #39
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']):
        with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'):
            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                result = run_pre_prompt_hook('test_repo')
                assert result == 'temp_repo'
                mock_run_script.assert_called_once_with('test_script.py', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']):
        with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'):
            with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('test error')):
                with pytest.raises(FailedHookException) as excinfo:
                    run_pre_prompt_hook('test_repo')
                assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.sh"
        hook_script.write_text("#!/bin/sh\nexit 0")

        context = {"test": "value"}

        run_hook_from_repo_dir(
            repo_dir,
            "post_gen_project",
            project_dir,
            context,
            delete_project_on_failure=True,
        )

        assert project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.sh"
        hook_script.write_text("#!/bin/sh\nexit 1")

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=True,
            )

        assert not project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.sh"
        hook_script.write_text("#!/bin/sh\nexit 1")

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=False,
            )

        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {"test": "value"}

        run_hook_from_repo_dir(
            repo_dir,
            "post_gen_project",
            project_dir,
            context,
            delete_project_on_failure=True,
        )

        assert project_dir.exists()


# LLM-generated content at query #41
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'test_key': 'test_value'}
    hook_name = 'pre_gen_project'
    hook_script = os.path.join(repo_dir, 'hooks', f'{hook_name}.py')
    os.makedirs(os.path.dirname(hook_script))
    with open(hook_script, 'w') as f:
        f.write('exit(0)')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert os.path.exists(project_dir)

    # Test failed execution with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('exit(1)')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Test failed execution with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"key": "value"}
    hook_name = "pre_gen_project"
    hook_dir = repo_dir / "hooks"
    hook_dir.mkdir()
    hook_file = hook_dir / f"{hook_name}.py"
    hook_file.write_text("print('test')")

    # Mock work_in context manager
    work_in_mock = mocker.patch("cookiecutter.hooks.work_in")
    work_in_mock.return_value.__enter__.return_value = repo_dir

    # Mock run_hook to avoid actual execution
    run_hook_mock = mocker.patch("cookiecutter.hooks.run_hook")

    # Test successful execution
    run_hook_from_repo_dir(
        repo_dir, hook_name, project_dir, context, delete_project_on_failure=True
    )
    run_hook_mock.assert_called_once_with(hook_name, project_dir, context)

    # Test FailedHookException
    run_hook_mock.side_effect = FailedHookException("test error")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure=True
        )
    assert not project_dir.exists()

    # Test UndefinedError
    run_hook_mock.side_effect = UndefinedError("test error")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure=False
        )
    assert project_dir.exists()


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/bash\necho "Hook executed"')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test hook failure with project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/bash\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )

            # Verify project directory was deleted
            assert not os.path.exists(project_dir)

    # Test hook failure without project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/bash\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )

            # Verify project directory still exists
            assert os.path.exists(project_dir)

    # Test with UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a hook script with undefined variable
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project.py')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('{{ undefined_variable }}')

            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )

            # Verify project directory was deleted
            assert not os.path.exists(project_dir)

    # Test with no hooks directory
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )
            # Should not raise any exception and project_dir should still exist
            assert os.path.exists(project_dir)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    hook_name = "pre_gen_project"
    context = {"project_name": "test_project"}
    delete_project_on_failure = True

    # Create a mock hook script
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / f"{hook_name}.py"
    hook_script.write_text("print('Hook executed')")

    # Mock the work_in context manager
    mocker.patch("cookiecutter.hooks.work_in", return_value=repo_dir)

    # Mock the run_hook function to avoid actual execution
    mocker.patch("cookiecutter.hooks.run_hook")

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert run_hook.called

    # Test FailedHookException
    run_hook.side_effect = FailedHookException("Hook failed")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert not project_dir.exists()

    # Test UndefinedError
    run_hook.side_effect = UndefinedError("Undefined variable")
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert not project_dir.exists()


# LLM-generated content at query #45
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test when pre_prompt hook is found and runs successfully
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

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/repo'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Hook failed')

        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #46
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        assert True  # Should not raise any exception

    # Test case 2: Hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/tmp/hook.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
            mock_run.assert_called_once_with('/tmp/hook.py', '/tmp/project', {'name': 'test'})

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook', return_value=['/tmp/hook1.py', '/tmp/hook2.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('post_gen_project', '/tmp/project', {'name': 'test'})
            assert mock_run.call_count == 2
            mock_run.assert_any_call('/tmp/hook1.py', '/tmp/project', {'name': 'test'})
            mock_run.assert_any_call('/tmp/hook2.py', '/tmp/project', {'name': 'test'})


# LLM-generated content at query #47
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        run_script(script_path, cwd=tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")\n')
        run_script(script_path, cwd=tmpdir)

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nnonexistent_command\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hook_dir = repo_dir / "hooks"
        hook_dir.mkdir()
        hook_file = hook_dir / "post_gen_project.sh"
        hook_file.write_text("#!/bin/sh\necho 'Hello'\n")

        context = {"test": "value"}
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()

    # Test failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hook_dir = repo_dir / "hooks"
        hook_dir.mkdir()
        hook_file = hook_dir / "post_gen_project.sh"
        hook_file.write_text("#!/bin/sh\nexit 1\n")

        context = {"test": "value"}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
        assert not project_dir.exists()

    # Test failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hook_dir = repo_dir / "hooks"
        hook_dir.mkdir()
        hook_file = hook_dir / "post_gen_project.sh"
        hook_file.write_text("#!/bin/sh\nexit 1\n")

        context = {"test": "value"}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()

    # Test with no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {"test": "value"}
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()


# LLM-generated content at query #49
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


# LLM-generated content at query #50
#--------------------------

```python
def test_run_pre_prompt_hook(mocker, tmp_path):
    # Mock the work_in context manager
    mock_work_in = mocker.patch('cookiecutter.hooks.work_in')

    # Mock find_hook to return None (no pre_prompt hook)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    result = run_pre_prompt_hook(tmp_path)
    assert result == tmp_path
    mock_work_in.assert_called_once_with(tmp_path)

    # Reset mocks
    mock_work_in.reset_mock()

    # Mock find_hook to return a script
    mock_script = tmp_path / 'pre_prompt.py'
    mock_script.write_text('#!/usr/bin/env python\nprint("test")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(mock_script)])

    # Mock create_tmp_repo_dir
    mock_tmp_dir = tmp_path / 'tmp'
    mock_tmp_dir.mkdir()
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=str(mock_tmp_dir))

    # Mock run_script
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')

    result = run_pre_prompt_hook(tmp_path)
    assert result == str(mock_tmp_dir)
    mock_work_in.assert_has_calls([mocker.call(tmp_path), mocker.call(str(mock_tmp_dir))])
    mock_run_script.assert_called_once_with(str(mock_script), str(mock_tmp_dir))

    # Test FailedHookException
    mock_run_script.side_effect = FailedHookException('test error')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(tmp_path)


# LLM-generated content at query #51
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)

        assert result == Path('/fake/tmp/repo')
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/fake/tmp/repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/fake/tmp/repo')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Hook failed')

        repo_dir = Path('/fake/repo')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook(repo_dir)

        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #52
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

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, False
            )

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script that fails
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
                )
            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script that fails
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, False
                )
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, False
            )


# LLM-generated content at query #53
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "post_gen_project.py"
    hook_file.write_text("print('Hook executed')")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    context = {"project_name": "test"}

    # Mock work_in to avoid changing directory
    mocker.patch('cookiecutter.utils.work_in', lambda x: x)

    # Test successful hook execution
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)

    # Verify hook was executed
    assert hook_file.exists()

    # Test hook failure with delete_project_on_failure=True
    hook_file.write_text("import sys; sys.exit(1)")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    assert not project_dir.exists()

    # Test hook failure with delete_project_on_failure=False
    project_dir.mkdir()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    assert project_dir.exists()

    # Test with non-existent hook
    run_hook_from_repo_dir(repo_dir, 'non_existent_hook', project_dir, context, True)
    assert project_dir.exists()


# LLM-generated content at query #54
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test when no hook is found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_debug = mocker.patch('cookiecutter.hooks.logger.debug')
    run_hook('pre_gen_project', tmp_path, {})
    logger_debug.assert_called_with('No %s hook found', 'pre_gen_project')

    # Test when hook is found and executed
    hook_script = tmp_path / 'hook.py'
    hook_script.write_text('print("Hello")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'name': 'test'})
    run_script_with_context.assert_called_with(hook_script, tmp_path, {'name': 'test'})

    # Test when multiple hooks are found
    hook_script2 = tmp_path / 'hook2.py'
    hook_script2.write_text('print("World")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script), str(hook_script2)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'name': 'test'})
    assert run_script_with_context.call_count == 2


# LLM-generated content at query #55
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

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('test error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #56
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\necho "test"\n')
        context = {'test': 'value'}

        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert project_dir.exists()

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\nexit 1\n')
        context = {'test': 'value'}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert not project_dir.exists()

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.sh'
        hook_file.write_text('#!/bin/sh\nexit 1\n')
        context = {'test': 'value'}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()
        context = {'test': 'value'}

        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert project_dir.exists()

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        project_dir = Path(tmpdir) / 'project'
        repo_dir.mkdir()
        project_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('{{ undefined_var }}\n')
        context = {'test': 'value'}

        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        assert not project_dir.exists()


# LLM-generated content at query #57
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Mock the find_hook function to return a list of scripts
    mocker.patch('cookiecutter.hooks.find_hook', return_value=['script1.py', 'script2.sh'])

    # Mock the run_script_with_context function to avoid actual execution
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Create a temporary directory for the project
    project_dir = tmp_path / 'test_project'
    project_dir.mkdir()

    # Define the context
    context = {'project_name': 'test_project'}

    # Call the function
    run_hook('pre_gen_project', project_dir, context)

    # Assert that run_script_with_context was called for each script
    assert mock_run_script.call_count == 2
    mock_run_script.assert_any_call('script1.py', project_dir, context)
    mock_run_script.assert_any_call('script2.sh', project_dir, context)

def test_run_hook_no_scripts(mocker, tmp_path, caplog):
    # Mock the find_hook function to return None (no scripts found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Mock the run_script_with_context function to avoid actual execution
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Create a temporary directory for the project
    project_dir = tmp_path / 'test_project'
    project_dir.mkdir()

    # Define the context
    context = {'project_name': 'test_project'}

    # Call the function
    with caplog.at_level(logging.DEBUG):
        run_hook('pre_gen_project', project_dir, context)

    # Assert that run_script_with_context was not called
    mock_run_script.assert_not_called()

    # Assert that a debug log was made
    assert 'No pre_gen_project hook found' in caplog.text


# LLM-generated content at query #58
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')

        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')

        run_script(script_path, cwd=tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test failure of a script with no shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test failure of a script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(0)')

        with mock.patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError(errno.ENOENT, 'File not found')
            with pytest.raises(FailedHookException) as excinfo:
                run_script(script_path, cwd=tmpdir)
            assert 'Hook script failed (error: File not found)' in str(excinfo.value)


# LLM-generated content at query #59
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #60
#--------------------------

```python
def test_run_hook():
    # Test case 1: Hook not found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')

    # Test case 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.sh']
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')
        mock_run_script.assert_called_once_with(
            '/tmp/hooks/pre_gen_project.sh',
            '/tmp/project',
            {'name': 'test'}
        )

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = [
            '/tmp/hooks/pre_gen_project.sh',
            '/tmp/hooks/pre_gen_project.py'
        ]
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')
        assert mock_run_script.call_count == 2
        mock_run_script.assert_any_call(
            '/tmp/hooks/pre_gen_project.sh',
            '/tmp/project',
            {'name': 'test'}
        )
        mock_run_script.assert_any_call(
            '/tmp/hooks/pre_gen_project.py',
            '/tmp/project',
            {'name': 'test'}
        )


# LLM-generated content at query #61
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
                f.write('print("Hook executed")')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, True
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

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
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

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, False
                )
            assert os.path.exists(project_dir)

    # Test with non-existent hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'non_existent_hook', project_dir, context, True
            )

    # Test with UndefinedError in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a hook script that will raise UndefinedError
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('{{ undefined_variable }}')

            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
                )
            assert not os.path.exists(project_dir)


# LLM-generated content at query #62
#--------------------------

```python
def test_run_script(mocker, tmp_path):
    # Test successful script execution
    script_path = tmp_path / 'test_script.sh'
    script_path.write_text('#!/bin/sh\necho "Hello, World!"\n')
    mocker.patch('subprocess.Popen')
    mocker.patch('cookiecutter.utils.make_executable')

    run_script(str(script_path))

    subprocess.Popen.assert_called_once_with(
        [str(script_path)],
        shell=False,
        cwd='.'
    )

    # Test failed script execution
    script_path = tmp_path / 'failing_script.sh'
    script_path.write_text('#!/bin/sh\nexit 1\n')
    mocker.patch('subprocess.Popen')
    mocker.patch('cookiecutter.utils.make_executable')
    proc_mock = mocker.MagicMock()
    proc_mock.wait.return_value = 1
    subprocess.Popen.return_value = proc_mock

    with pytest.raises(FailedHookException) as excinfo:
        run_script(str(script_path))
    assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test Python script execution
    script_path = tmp_path / 'test_python.py'
    script_path.write_text('print("Hello from Python")\n')
    mocker.patch('subprocess.Popen')
    mocker.patch('cookiecutter.utils.make_executable')

    run_script(str(script_path))

    subprocess.Popen.assert_called_once_with(
        [sys.executable, str(script_path)],
        shell=False,
        cwd='.'
    )

    # Test Windows platform
    mocker.patch('sys.platform', 'win32')
    script_path = tmp_path / 'test_script.bat'
    script_path.write_text('echo Hello from Windows\n')
    mocker.patch('subprocess.Popen')
    mocker.patch('cookiecutter.utils.make_executable')

    run_script(str(script_path))

    subprocess.Popen.assert_called_once_with(
        [str(script_path)],
        shell=True,
        cwd='.'
    )

    # Test OSError with ENOEXEC
    script_path = tmp_path / 'empty_script.sh'
    script_path.write_text('')
    mocker.patch('subprocess.Popen', side_effect=OSError(errno.ENOEXEC, 'No exec'))

    with pytest.raises(FailedHookException) as excinfo:
        run_script(str(script_path))
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test other OSError
    mocker.patch('subprocess.Popen', side_effect=OSError(errno.EACCES, 'Permission denied'))

    with pytest.raises(FailedHookException) as excinfo:
        run_script(str(script_path))
    assert 'Hook script failed (error: [Errno 13] Permission denied)' in str(excinfo.value)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/repo'
        mock_run_script.assert_called_once_with('/fake/hook.py', '/tmp/repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/tmp/repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #2
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
                f.write('print("Hook executed")')

            # Run the hook
            context = {'test': 'value'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
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

            # Run the hook and expect failure
            context = {'test': 'value'}
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

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect failure
            context = {'test': 'value'}
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

    # Test with non-existent hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'test': 'value'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='non_existent_hook',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
            # Should not raise any exception


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')

        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')

        run_script(script_path, cwd=tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test failure of a script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')

        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd=tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)


# LLM-generated content at query #4
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


# LLM-generated content at query #5
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #6
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test case 3: Hook directory exists with valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 5: Hook directory exists with backup file (~)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 6: Hook directory exists with multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook():
    # Test valid hook file
    assert valid_hook('pre_gen_project.py', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.sh', 'post_gen_project') is True

    # Test invalid hook file (wrong name)
    assert valid_hook('pre_gen_project.py', 'post_gen_project') is False
    assert valid_hook('invalid_hook.py', 'pre_gen_project') is False

    # Test invalid hook file (backup file)
    assert valid_hook('pre_gen_project.py~', 'pre_gen_project') is False

    # Test invalid hook file (unsupported hook)
    assert valid_hook('invalid_hook.py', 'invalid_hook') is False


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook(mocker, tmp_path):
    # Test case 1: No pre_prompt hook found
    repo_dir = tmp_path / "no_hook"
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

    # Test case 2: Valid pre_prompt hook found and executed successfully
    repo_dir = tmp_path / "with_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('Pre-prompt hook executed')")

    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    mocker.patch('cookiecutter.hooks.work_in', return_value=repo_dir)

    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

    # Test case 3: Pre_prompt hook fails
    repo_dir = tmp_path / "failing_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")

    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir)
    mocker.patch('cookiecutter.hooks.work_in', return_value=repo_dir)

    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        # Create a valid hook script
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("print('Hook executed')")

        context = {"project_name": "test"}

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )

        assert project_dir.exists()

    # Test hook failure with project deletion
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        # Create a failing hook script
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("import sys; sys.exit(1)")

        context = {"project_name": "test"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="post_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True,
            )

        assert not project_dir.exists()

    # Test hook failure without project deletion
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        # Create a failing hook script
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("import sys; sys.exit(1)")

        context = {"project_name": "test"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="post_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False,
            )

        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {"project_name": "test"}

        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )

        assert project_dir.exists()


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script():
    # Create a temporary directory and script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/bash\necho "Hello, World!"\n')

        # Test successful script execution
        run_script(str(script_path), cwd=tmpdir)

        # Test script with non-zero exit status
        bad_script_path = Path(tmpdir) / 'bad_script.sh'
        bad_script_path.write_text('#!/bin/bash\nexit 1\n')
        with pytest.raises(FailedHookException):
            run_script(str(bad_script_path), cwd=tmpdir)

        # Test script with missing shebang
        no_shebang_script = Path(tmpdir) / 'no_shebang.sh'
        no_shebang_script.write_text('echo "No shebang"\n')
        with pytest.raises(FailedHookException):
            run_script(str(no_shebang_script), cwd=tmpdir)

        # Test Python script execution
        py_script_path = Path(tmpdir) / 'test_script.py'
        py_script_path.write_text('print("Hello from Python")\n')
        run_script(str(py_script_path), cwd=tmpdir)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_script_with_context():
    # Test with a simple script
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')

    context = {'name': 'World'}
    cwd = tempfile.mkdtemp()

    run_script_with_context(script_path, cwd, context)

    # Cleanup
    script_path.unlink()
    rmtree(cwd)

    # Test with a non-Python script
    script_content = "echo 'Hello, {{ name }}!'"
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text(script_content, encoding='utf-8')

    context = {'name': 'World'}
    cwd = tempfile.mkdtemp()

    run_script_with_context(script_path, cwd, context)

    # Cleanup
    script_path.unlink()
    rmtree(cwd)

    # Test with a script that fails
    script_content = "exit 1"
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text(script_content, encoding='utf-8')

    context = {}
    cwd = tempfile.mkdtemp()

    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, cwd, context)

    # Cleanup
    script_path.unlink()
    rmtree(cwd)


# LLM-generated content at query #5
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
    script_path.unlink()
    os.rmdir(cwd)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('hook_script', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook():
    # Test valid hook
    assert valid_hook('pre_gen_project.py', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.sh', 'post_gen_project') is True

    # Test invalid hook names
    assert valid_hook('invalid_hook.py', 'pre_gen_project') is False
    assert valid_hook('pre_gen_project.py', 'invalid_hook') is False

    # Test unsupported hook
    assert valid_hook('invalid_hook.py', 'invalid_hook') is False

    # Test backup file
    assert valid_hook('pre_gen_project.py~', 'pre_gen_project') is False


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    # Setup
    repo_dir = tmp_path / 'repo'
    project_dir = tmp_path / 'project'
    context = {'test': 'value'}
    hook_name = 'pre_gen_project'
    hook_script = repo_dir / 'hooks' / f'{hook_name}.py'
    hook_script.parent.mkdir(parents=True)
    hook_script.write_text('print("Hook executed")')

    # Mocks
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.run_hook')
    mocker.patch('cookiecutter.hooks.rmtree')
    mocker.patch('cookiecutter.hooks.logger.exception')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    cookiecutter.hooks.run_hook.assert_called_once_with(hook_name, project_dir, context)

    # Test failed execution with delete_project_on_failure=True
    cookiecutter.hooks.run_hook.side_effect = FailedHookException('Hook failed')
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    cookiecutter.hooks.rmtree.assert_called_once_with(project_dir)
    cookiecutter.hooks.logger.exception.assert_called_once()

    # Test failed execution with delete_project_on_failure=False
    cookiecutter.hooks.run_hook.side_effect = FailedHookException('Hook failed')
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    cookiecutter.hooks.rmtree.assert_not_called()
    cookiecutter.hooks.logger.exception.assert_called()


# LLM-generated content at query #9
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

    # Test no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', os.path.join(tmpdir, 'nonexistent'))
        assert result is None

    # Test no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'other_hook.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test multiple matching hooks
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

    # Cleanup
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("print('Hook executed')")

        context = {"test": "value"}

        run_hook_from_repo_dir(
            repo_dir,
            "post_gen_project",
            project_dir,
            context,
            delete_project_on_failure=False,
        )

        assert project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("import sys; sys.exit(1)")

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=True,
            )

        assert not project_dir.exists()

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("import sys; sys.exit(1)")

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=False,
            )

        assert project_dir.exists()

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()

        hook_script = hooks_dir / "post_gen_project.py"
        hook_script.write_text("{{ undefined_variable }}")

        context = {"test": "value"}

        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=True,
            )

        assert not project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {"test": "value"}

        run_hook_from_repo_dir(
            repo_dir,
            "post_gen_project",
            project_dir,
            context,
            delete_project_on_failure=False,
        )

        assert project_dir.exists()


# LLM-generated content at query #12
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_prompt', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_prompt', tmpdir) is None

    # Test case 3: Hook directory exists with valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_prompt.py')
        Path(hook_file).touch()
        result = find_hook('pre_prompt', hook_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_prompt', hook_dir) is None

    # Test case 5: Hook directory exists with backup file (~)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_prompt.py~')
        Path(hook_file).touch()
        assert find_hook('pre_prompt', hook_dir) is None

    # Test case 6: Hook directory exists with multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_prompt.py')
        hook_file2 = os.path.join(hook_dir, 'pre_prompt.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_prompt', hook_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result


# LLM-generated content at query #13
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
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp_repo_dir.return_value = '/fake/tmp_repo_dir'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/tmp_repo_dir'
        mock_run_script.assert_called_once_with('/fake/script.py', '/fake/tmp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp_repo_dir.return_value = '/fake/tmp_repo_dir'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('/fake/repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'project_name': 'test_project'}
    hook_name = 'pre_gen_project'

    # Create a test hook script
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, f'{hook_name}.py')
    with open(hook_script, 'w') as f:
        f.write('print("Hook executed")')

    # Test successful hook execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    assert os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('script.sh', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('test_repo')


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "post_gen_project.py").write_text("print('success')")

        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        context = {"test": "value"}

        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)

        assert project_dir.exists()

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "post_gen_project.py").write_text("import sys; sys.exit(1)")

        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)

        assert not project_dir.exists()

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "post_gen_project.py").write_text("import sys; sys.exit(1)")

        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        context = {"test": "value"}

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)

        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()

        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        context = {"test": "value"}

        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)

        assert project_dir.exists()

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "post_gen_project.py").write_text("{{ undefined_var }}")

        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        context = {"test": "value"}

        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)

        assert not project_dir.exists()


# LLM-generated content at query #17
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

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test_project'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #18
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])

    # Mock run_script_with_context
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert run_script_with_context.called
    run_script_with_context.assert_called_once_with(str(script_path), project_dir, context)

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
    assert not run_script_with_context.called
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook exists
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook exists and runs successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('hook_script', 'temp_repo')

    # Test case 3: Pre_prompt hook exists but fails
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
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

    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('script.sh', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hooks_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hooks_dir)
            hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir, 'post_gen_project', project_dir, context, False
                )
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir, 'post_gen_project', project_dir, context, True
            )


# LLM-generated content at query #24
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test case 3: Hook directory exists with valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 5: Hook directory exists with backup file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 6: Hook directory exists with multiple valid hooks
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


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/fake/tmp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/tmp_repo'
        mock_run_script.assert_called_once_with('/fake/script.py', '/fake/tmp_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/fake/tmp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('/fake/repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            # Run the hook
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context={},
                delete_project_on_failure=False,
            )

            # Verify the hook was executed (by checking if the project_dir still exists)
            assert os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script that fails
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect it to fail
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context={},
                    delete_project_on_failure=True,
                )

            # Verify the project_dir was deleted
            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script that fails
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect it to fail
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context={},
                    delete_project_on_failure=False,
                )

            # Verify the project_dir still exists
            assert os.path.exists(project_dir)

    # Test with no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Run the hook and expect it to do nothing
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context={},
                delete_project_on_failure=False,
            )

            # Verify the project_dir still exists
            assert os.path.exists(project_dir)


# LLM-generated content at query #28
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

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create invalid hook files
        invalid_hook_file1 = os.path.join(hooks_dir, 'invalid_hook.py')
        invalid_hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(invalid_hook_file1).touch()
        Path(invalid_hook_file2).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test with non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent_hooks_dir')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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


# LLM-generated content at query #31
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test case 3: Hook directory exists with matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with multiple matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert set(result) == {hook_file1, hook_file2}

    # Test case 5: Hook directory exists with invalid hooks (backup files)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 6: Hook directory exists with non-matching hook names
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script_with_context():
    # Setup test data
    script_content = "print('Hello, {{ name }}!')"
    context = {'name': 'World'}
    extension = '.py'

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as temp_script:
        temp_script.write(script_content)
        temp_script_path = temp_script.name

    # Create a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Call the function
            run_script_with_context(temp_script_path, temp_dir, context)

            # Verify the script was executed successfully
            assert True
        finally:
            # Clean up the temporary script file
            os.unlink(temp_script_path)


# LLM-generated content at query #34
#--------------------------

```python
def test_run_hook():
    # Test when no hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        run_hook('pre_gen_project', '/fake/project', {'key': 'value'})
        assert True  # No exception should be raised

    # Test when hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/fake/project', {'key': 'value'})
            mock_run.assert_called_once_with('/fake/hook.py', '/fake/project', {'key': 'value'})

    # Test when multiple hooks are found and executed
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook1.py', '/fake/hook2.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/fake/project', {'key': 'value'})
            assert mock_run.call_count == 2
            mock_run.assert_any_call('/fake/hook1.py', '/fake/project', {'key': 'value'})
            mock_run.assert_any_call('/fake/hook2.py', '/fake/project', {'key': 'value'})


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a simple hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'pre_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'pre_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'pre_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'pre_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #36
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello"\n')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test script with non-zero exit status
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\nexit 1\n')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test Python script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello")\n')
        f.flush()
        script_path = f.name

    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test script with OSError (missing shebang)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('echo "Hello"\n')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
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
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a simple hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            # Verify project directory was deleted
            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            # Verify project directory was not deleted
            assert os.path.exists(project_dir)

    # Test with non-existent hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'non_existent_hook',
                project_dir,
                context,
                delete_project_on_failure=True
            )
            # Should not raise any exception and project dir should exist
            assert os.path.exists(project_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'project_name': 'test'}

    # Create a hook script
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_script, 'w') as f:
        f.write('print("Hook executed")')

    # Test successful execution
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    # Test failure with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
        pytest.fail("Expected FailedHookException")
    except FailedHookException:
        assert not os.path.exists(project_dir)

    # Test failure with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    try:
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
        pytest.fail("Expected FailedHookException")
    except FailedHookException:
        assert os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_script_with_context():
    # Test successful execution of a script with context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, {{ name }}!")', encoding='utf-8')

        context = {'name': 'World'}

        run_script_with_context(script_path, tmpdir, context)

    # Test execution of a non-Python script with context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('echo "Hello, {{ name }}!"', encoding='utf-8')

        context = {'name': 'World'}

        run_script_with_context(script_path, tmpdir, context)

    # Test execution of a script that fails
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('import sys; sys.exit(1)', encoding='utf-8')

        context = {}

        with pytest.raises(FailedHookException):
            run_script_with_context(script_path, tmpdir, context)

    # Test execution of a script with undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, {{ undefined_var }}!")', encoding='utf-8')

        context = {}

        with pytest.raises(UndefinedError):
            run_script_with_context(script_path, tmpdir, context)


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\necho "Hello, World!"\n')
        script_path = f.name

    try:
        run_script(script_path)
        assert True
    finally:
        os.unlink(script_path)

    # Test failed script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('#!/bin/sh\nexit 1\n')
        script_path = f.name

    try:
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert 'Hook script failed (exit status: 1)' in str(exc_info.value)
    finally:
        os.unlink(script_path)

    # Test Python script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("Hello, World!")\n')
        script_path = f.name

    try:
        run_script(script_path)
        assert True
    finally:
        os.unlink(script_path)

    # Test non-existent script
    with pytest.raises(FailedHookException) as exc_info:
        run_script('/non/existent/script.sh')
    assert 'Hook script failed' in str(exc_info.value)

    # Test empty script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('')
        script_path = f.name

    try:
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(exc_info.value)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #42
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

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

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

        # Create a backup hook file
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Backup hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a hook file with wrong name
        wrong_hook_file = os.path.join(hooks_dir, 'wrong_hook.py')
        with open(wrong_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Wrong hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent_hooks_dir')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #43
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.py']):
        with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=Path('/fake/tmp/repo')):
            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                repo_dir = Path('/fake/repo')
                result = run_pre_prompt_hook(repo_dir)
                assert result == Path('/fake/tmp/repo')
                mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/fake/tmp/repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/repo/hooks/pre_prompt.py']):
        with patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=Path('/fake/tmp/repo')):
            with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Hook failed')):
                repo_dir = Path('/fake/repo')
                with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
                    run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a dummy hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            # Run the hook
            context = {'test': 'value'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='pre_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

    # Test hook failure with project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect failure
            context = {'test': 'value'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='pre_gen_project',
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
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'pre_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            # Run the hook and expect failure
            context = {'test': 'value'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='pre_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )

            # Verify project directory still exists
            assert os.path.exists(project_dir)

    # Test with no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Run the hook with no hook found
            context = {'test': 'value'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='pre_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

            # Verify project directory still exists
            assert os.path.exists(project_dir)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        with patch('cookiecutter.hooks.logger.debug') as mock_debug:
            run_hook('pre_gen_project', '/fake/project_dir', {})
            mock_debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook_script.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('pre_gen_project', '/fake/project_dir', {'key': 'value'})
            mock_run_script.assert_called_once_with('/fake/hook_script.py', '/fake/project_dir', {'key': 'value'})

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/hook_script1.py', '/fake/hook_script2.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('post_gen_project', '/fake/project_dir', {'key': 'value'})
            assert mock_run_script.call_count == 2
            mock_run_script.assert_any_call('/fake/hook_script1.py', '/fake/project_dir', {'key': 'value'})
            mock_run_script.assert_any_call('/fake/hook_script2.py', '/fake/project_dir', {'key': 'value'})


# LLM-generated content at query #46
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        project_dir = Path(temp_dir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        # Create a valid hook script
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "post_gen_project.sh"
        hook_script.write_text("#!/bin/sh\necho 'Hello'\n")

        context = {"project_name": "test"}

        # Test successful execution
        run_hook_from_repo_dir(
            repo_dir,
            "post_gen_project",
            project_dir,
            context,
            delete_project_on_failure=True,
        )

        # Verify project_dir still exists
        assert project_dir.exists()

        # Test failure with delete_project_on_failure=True
        failing_hook = hooks_dir / "post_gen_project_fail.sh"
        failing_hook.write_text("#!/bin/sh\nexit 1\n")

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=True,
            )

        # Verify project_dir was deleted
        assert not project_dir.exists()

        # Recreate project_dir for next test
        project_dir.mkdir()

        # Test failure with delete_project_on_failure=False
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                "post_gen_project",
                project_dir,
                context,
                delete_project_on_failure=False,
            )

        # Verify project_dir still exists
        assert project_dir.exists()


# LLM-generated content at query #47
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

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Test error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('script.py', 'temp_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #49
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
        run_script('/nonexistent/script.py')


# LLM-generated content at query #50
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
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)
    assert cookiecutter.hooks.run_hook.called

    # Reset mocks
    cookiecutter.hooks.run_hook.reset_mock()
    cookiecutter.hooks.rmtree.reset_mock()

    # Test failed execution with delete_project_on_failure=True
    cookiecutter.hooks.run_hook.side_effect = FailedHookException("Test error")
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)
    assert cookiecutter.hooks.rmtree.called

    # Reset mocks
    cookiecutter.hooks.run_hook.reset_mock()
    cookiecutter.hooks.rmtree.reset_mock()

    # Test failed execution with delete_project_on_failure=False
    delete_project_on_failure = False
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, delete_project_on_failure)
    assert not cookiecutter.hooks.rmtree.called


# LLM-generated content at query #51
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    with pytest.raises(FailedHookException):
        find_hook('pre_gen_project', 'non_existent_dir')

    # Test case 2: Hook directory exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test case 3: Hook directory exists with valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert result == [hook_file]

    # Test case 4: Hook directory exists with invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 5: Hook directory exists with backup file
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 6: Hook directory exists with multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', hook_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result


# LLM-generated content at query #52
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        os.chmod(script_path, 0o755)
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

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")\n')
        run_script(script_path, tmpdir)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"\n')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test non-existent script
    with pytest.raises(FailedHookException) as excinfo:
        run_script('/nonexistent/script.sh', '.')
    assert 'Hook script failed (error: [Errno 2] No such file or directory' in str(excinfo.value)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')

    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    find_hook_mock = mocker.patch('cookiecutter.hooks.find_hook')
    find_hook_mock.assert_called_once_with(hook_name)
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_script_with_context_mock.assert_called_once_with(str(script_path), project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'non_existent_hook'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text
    run_script_with_context_mock = mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_script_with_context_mock.assert_not_called()


# LLM-generated content at query #54
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

    # Test multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook1).touch()
        Path(hook2).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook1 in result
        assert hook2 in result


# LLM-generated content at query #55
#--------------------------

```python
def test_find_hook():
    # Test when hooks_dir does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test when hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test when hooks_dir exists with non-matching hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'invalid_hook.py'), 'w') as f:
            f.write('# invalid hook')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test when hooks_dir exists with matching hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('# valid hook')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0].endswith('pre_gen_project.py')

    # Test when hooks_dir exists with multiple matching hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('# valid hook 1')
        with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
            f.write('# valid hook 2')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert any(script.endswith('pre_gen_project.py') for script in result)
        assert any(script.endswith('pre_gen_project.sh') for script in result)

    # Test when hooks_dir exists with backup files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
            f.write('# backup hook')
        assert find_hook('pre_gen_project', hooks_dir) is None


# LLM-generated content at query #56
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #57
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
    os.unlink(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #58
#--------------------------

```python
def test_find_hook():
    # Test when hooks_dir does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test when hooks_dir exists but is empty
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test when hooks_dir contains non-matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'invalid_hook.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test when hooks_dir contains a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.join(hooks_dir, 'pre_gen_project.py')

    # Test when hooks_dir contains multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.join(hooks_dir, 'pre_gen_project.py') in result
        assert os.path.join(hooks_dir, 'pre_gen_project.sh') in result

    # Test when hooks_dir contains backup files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')
        assert find_hook('pre_gen_project', hooks_dir) is None


# LLM-generated content at query #59
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


# LLM-generated content at query #60
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/tmp/fake_repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/fake_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = lambda self: self
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Hook failed')

        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('/fake/repo')

        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #61
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "post_gen_project.py"
        hook_file.write_text("print('Hook executed')")

        context = {"test": "value"}
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()

    # Test hook failure with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "post_gen_project.py"
        hook_file.write_text("import sys; sys.exit(1)")

        context = {"test": "value"}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
        assert not project_dir.exists()

    # Test hook failure with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "post_gen_project.py"
        hook_file.write_text("import sys; sys.exit(1)")

        context = {"test": "value"}
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()

    # Test no hook found
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        context = {"test": "value"}
        run_hook_from_repo_dir(repo_dir, "nonexistent_hook", project_dir, context, False)
        assert project_dir.exists()

    # Test UndefinedError in hook
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        project_dir = Path(tmpdir) / "project"
        repo_dir.mkdir()
        project_dir.mkdir()

        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "post_gen_project.py"
        hook_file.write_text("{{ undefined_variable }}")

        context = {"test": "value"}
        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
        assert project_dir.exists()


# LLM-generated content at query #62
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

    # Test with non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #63
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')

    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert run_script_with_context.called
    run_script_with_context.assert_called_once_with(str(script_path), project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}

    # Mock find_hook to return None (no hook found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert not run_script_with_context.called
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #64
#--------------------------

```python
def test_run_hook():
    # Test when no hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')

    # Test when hook is found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/tmp/hook_script.py']
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('pre_gen_project')
        mock_run_script.assert_called_once_with('/tmp/hook_script.py', '/tmp/project', {'name': 'test'})

    # Test when multiple hooks are found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        mock_find_hook.return_value = ['/tmp/hook1.py', '/tmp/hook2.sh']
        run_hook('post_gen_project', '/tmp/project', {'name': 'test'})
        mock_find_hook.assert_called_once_with('post_gen_project')
        assert mock_run_script.call_count == 2
        mock_run_script.assert_any_call('/tmp/hook1.py', '/tmp/project', {'name': 'test'})
        mock_run_script.assert_any_call('/tmp/hook2.sh', '/tmp/project', {'name': 'test'})


# LLM-generated content at query #65
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('hook_script.py', 'temp_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook', return_value=['hook_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('test_repo')
        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'


# LLM-generated content at query #66
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('dummy_repo_dir')
        assert result == 'dummy_repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['dummy_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('dummy_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('dummy_script.py', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['dummy_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('dummy_repo_dir')


# LLM-generated content at query #67
#--------------------------

```python
def test_run_pre_prompt_hook(mocker, tmp_path):
    # Test case 1: No pre_prompt hook found
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

    # Test case 2: Pre_prompt hook found and executed successfully
    repo_dir = tmp_path / "test_repo_with_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("print('Hello, World!')")

    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_file)])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=str(repo_dir))
    mocker.patch('cookiecutter.hooks.run_script')

    result = run_pre_prompt_hook(repo_dir)
    assert result == str(repo_dir)

    # Test case 3: Pre_prompt hook found but execution failed
    repo_dir = tmp_path / "test_repo_with_failing_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_file = hooks_dir / "pre_prompt.py"
    hook_file.write_text("exit(1)")

    mocker.patch('cookiecutter.hooks.work_in')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_file)])
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=str(repo_dir))
    mocker.patch('cookiecutter.hooks.run_script', side_effect=FailedHookException("Hook script failed"))

    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #68
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

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [hook_file]

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding multiple hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result

    # Test finding invalid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create invalid hook files
        invalid_hook1 = os.path.join(hooks_dir, 'invalid_hook.py')
        invalid_hook2 = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(invalid_hook1).touch()
        Path(invalid_hook2).touch()

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent_hooks_dir')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #69
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    with pytest.raises(FailedHookException):
        find_hook('pre_gen_project', 'nonexistent_dir')

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
        assert set(result) == {hook_file1, hook_file2}

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


# LLM-generated content at query #70
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('fake_repo_dir')
        assert result == 'fake_repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['fake_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('fake_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('fake_script', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['fake_script']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('fake_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #71
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    result = find_hook('pre_gen_project', 'nonexistent_dir')
    assert result is None

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'other_script.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test case 3: Valid hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [os.path.join(hooks_dir, 'pre_gen_project.py')]

    # Test case 4: Multiple valid hooks exist
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
            f.write('#!/bin/sh\necho "Hello"')
        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert os.path.join(hooks_dir, 'pre_gen_project.py') in result
        assert os.path.join(hooks_dir, 'pre_gen_project.sh') in result

    # Test case 5: Backup file should be ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #72
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook exists
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            mock_find_hook.return_value = None
            result = run_pre_prompt_hook('/fake/repo')
            assert result == '/fake/repo'
            mock_find_hook.assert_called_once_with('pre_prompt')

    # Test when pre_prompt hook exists and succeeds
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            with patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp:
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
                    mock_create_tmp.return_value = '/tmp/repo'
                    result = run_pre_prompt_hook('/fake/repo')
                    assert result == '/tmp/repo'
                    mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/repo')

    # Test when pre_prompt hook exists but fails
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
            with patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp:
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
                    mock_create_tmp.return_value = '/tmp/repo'
                    mock_run_script.side_effect = FailedHookException('test error')
                    with pytest.raises(FailedHookException) as excinfo:
                        run_pre_prompt_hook('/fake/repo')
                    assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #73
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\necho "Hello, World!"\n')
        run_script(str(script_path), tmpdir)
        assert True  # If no exception is raised, the test passes

    # Test script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\nexit 1\n')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(str(script_path), tmpdir)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

    # Test script with missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('echo "Hello, World!"\n')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(str(script_path), tmpdir)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, World!")\n')
        run_script(str(script_path), tmpdir)
        assert True  # If no exception is raised, the test passes


# LLM-generated content at query #74
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('script.sh', 'temp_repo')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['script.sh']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook('test_repo')
        assert 'Pre-Prompt Hook script failed' in str(exc_info.value)


# LLM-generated content at query #75
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook]

    # Test finding no hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file (backup)
        invalid_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(invalid_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        valid_hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook1, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test1")')

        valid_hook2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(valid_hook2, 'w') as f:
            f.write('#!/bin/sh\necho "test2"')

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert valid_hook1 in result
        assert valid_hook2 in result

    # Test non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #76
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    context = {'project_name': 'test_project'}
    script_content = 'echo "Hello {{ project_name }}"'
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Cleanup
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #77
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello"\n')
        run_script(script_path, tmpdir)

    # Test failed script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello")\n')
        run_script(script_path, tmpdir)

    # Test missing shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello"\n')
        with pytest.raises(FailedHookException):
            run_script(script_path, tmpdir)


# LLM-generated content at query #78
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
    assert run_script_with_context.called
    run_script_with_context.assert_called_once_with(str(script_path), project_dir, context)

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
    assert not run_script_with_context.called
    assert f'No {hook_name} hook found' in caplog.text


# LLM-generated content at query #79
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #80
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo')
        assert result == 'test_repo'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['pre_prompt_script'], ['pre_prompt_script']]
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        result = run_pre_prompt_hook('test_repo')
        assert result == 'temp_repo'
        mock_run_script.assert_called_once_with('pre_prompt_script', 'temp_repo')

    # Test case 3: Pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.side_effect = [['pre_prompt_script'], ['pre_prompt_script']]
        mock_create_tmp.return_value = 'temp_repo'
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Test error')

        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('test_repo')


# LLM-generated content at query #81
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'test_repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['test_script']
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('test_script', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['test_script']
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        mock_run_script.side_effect = FailedHookException('Test error')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('test_repo_dir')


# LLM-generated content at query #82
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

    # Test failed execution of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test execution of a non-existent script
    with pytest.raises(FailedHookException):
        run_script('/non/existent/script.py')

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


# LLM-generated content at query #83
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'test_repo_dir'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo_dir')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['test_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #84
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        repo_dir = '/fake/repo'
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_work_in.return_value.__exit__ = mock_work_in

        repo_dir = '/fake/repo'
        result = run_pre_prompt_hook(repo_dir)

        assert result == '/tmp/fake_repo'
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/tmp/fake_repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = '/tmp/fake_repo'
        mock_work_in.return_value.__enter__ = mock_work_in
        mock_work_in.return_value.__exit__ = mock_work_in
        mock_run_script.side_effect = FailedHookException('Script failed')

        repo_dir = '/fake/repo'
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook(repo_dir)

        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #85
#--------------------------

```python
def test_run_hook():
    # Test when no hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
        assert True  # No exception should be raised

    # Test when hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/tmp/hook.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/tmp/project', {'name': 'test'})
            mock_run.assert_called_once_with('/tmp/hook.py', '/tmp/project', {'name': 'test'})

    # Test when multiple hooks are found and executed
    with patch('cookiecutter.hooks.find_hook', return_value=['/tmp/hook1.py', '/tmp/hook2.sh']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('post_gen_project', '/tmp/project', {'name': 'test'})
            assert mock_run.call_count == 2
            mock_run.assert_any_call('/tmp/hook1.py', '/tmp/project', {'name': 'test'})
            mock_run.assert_any_call('/tmp/hook2.sh', '/tmp/project', {'name': 'test'})


# LLM-generated content at query #86
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'project_name': 'test_project'}

    # Create a valid hook script
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_script = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_script, 'w') as f:
        f.write('print("Hook executed")')

    # Test successful execution
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    assert os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    assert not os.path.exists(project_dir)

    # Test hook failure with delete_project_on_failure=False
    project_dir = tempfile.mkdtemp()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    assert os.path.exists(project_dir)

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #87
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a simple hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\necho "Hook executed"')

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test hook failure with project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure without project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_script = os.path.join(repo_dir, 'hooks', 'post_gen_project')
            os.makedirs(os.path.dirname(hook_script))
            with open(hook_script, 'w') as f:
                f.write('#!/bin/sh\nexit 1')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test with no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #88
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


# LLM-generated content at query #89
#--------------------------

```python
def test_run_script():
    # Test successful script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 0')
        run_script(script_path, cwd=tmpdir)

    # Test failed script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\nexit 1')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('exit(0)')
        run_script(script_path, cwd=tmpdir)

    # Test non-existent script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'nonexistent.sh')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)


# LLM-generated content at query #90
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'test_repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['test_script.py']
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        result = run_pre_prompt_hook('test_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('test_script.py', 'temp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['test_script.py']
        mock_create_tmp_repo_dir.return_value = 'temp_repo_dir'
        mock_run_script.side_effect = FailedHookException('Test error')
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook('test_repo_dir')
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)


# LLM-generated content at query #91
#--------------------------

```python
def test_find_hook():
    # Test when hooks_dir does not exist
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)

        # Test when hooks_dir is empty
        assert find_hook('pre_gen_project', hooks_dir) is None

        # Create a valid hook file
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')

        # Test finding a valid hook
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(valid_hook_file)

        # Create an invalid hook file (wrong name)
        invalid_hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(invalid_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')

        # Test that invalid hook is not found
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(valid_hook_file)

        # Create a backup file (should be ignored)
        backup_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Hello")')

        # Test that backup file is ignored
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(valid_hook_file)

        # Test finding multiple valid hooks
        another_valid_hook = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(another_valid_hook, 'w') as f:
            f.write('#!/bin/sh\necho "Hello"')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert os.path.abspath(valid_hook_file) in result
        assert os.path.abspath(another_valid_hook) in result


# LLM-generated content at query #92
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_path, 'w') as f:
                f.write('#!/bin/sh\necho "Success"\n')
            utils.make_executable(hook_path)

            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )

    # Test failed hook execution with delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_path, 'w') as f:
                f.write('#!/bin/sh\nexit 1\n')
            utils.make_executable(hook_path)

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test failed hook execution with delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.sh')
            with open(hook_path, 'w') as f:
                f.write('#!/bin/sh\nexit 1\n')
            utils.make_executable(hook_path)

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test hook with UndefinedError
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_path = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_path, 'w') as f:
                f.write('print("{{ undefined_var }}")')

            context = {'project_name': 'test'}
            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir,
                    'post_gen_project',
                    project_dir,
                    context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir,
                'post_gen_project',
                project_dir,
                context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #93
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

    # Test failure of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test non-existent script
    with pytest.raises(FailedHookException):
        run_script('/non/existent/script.py')

    # Test empty script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #94
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test case 1: No hook found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    mocker.patch('cookiecutter.hooks.logger.debug')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert cookiecutter.hooks.logger.debug.called_once_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed
    script_path = tmp_path / 'hooks' / 'pre_gen_project.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    cookiecutter.hooks.run_script_with_context.assert_called_once_with(str(script_path), tmp_path, {'project_name': 'test'})

    # Test case 3: Multiple hooks found and executed
    script_path1 = tmp_path / 'hooks' / 'pre_gen_project.py'
    script_path1.parent.mkdir()
    script_path1.write_text('print("Hook 1 executed")')
    script_path2 = tmp_path / 'hooks' / 'pre_gen_project.sh'
    script_path2.write_text('echo "Hook 2 executed"')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path1), str(script_path2)])
    mocker.patch('cookiecutter.hooks.run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert cookiecutter.hooks.run_script_with_context.call_count == 2
    cookiecutter.hooks.run_script_with_context.assert_any_call(str(script_path1), tmp_path, {'project_name': 'test'})
    cookiecutter.hooks.run_script_with_context.assert_any_call(str(script_path2), tmp_path, {'project_name': 'test'})


# LLM-generated content at query #95
#--------------------------

```python
def test_run_script_with_context():
    # Setup test context
    context = {'project_name': 'test_project', 'author': 'test_author'}

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test script with Jinja2 template
        script_path = Path(tmpdir) / 'test_script.sh'
        script_content = """#!/bin/bash
echo "Project: {{ project_name }}"
echo "Author: {{ author }}"
"""
        script_path.write_text(script_content, encoding='utf-8')

        # Run the script with context
        run_script_with_context(script_path, tmpdir, context)

        # Verify the script was executed by checking the output
        output_file = Path(tmpdir) / 'output.txt'
        assert output_file.exists()
        assert 'Project: test_project' in output_file.read_text()
        assert 'Author: test_author' in output_file.read_text()

    # Test with Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_content = """print("Project: {{ project_name }}")
print("Author: {{ author }}")
"""
        script_path.write_text(script_content, encoding='utf-8')

        # Run the script with context
        run_script_with_context(script_path, tmpdir, context)

        # Verify the script was executed by checking the output
        output_file = Path(tmpdir) / 'output.txt'
        assert output_file.exists()
        assert 'Project: test_project' in output_file.read_text()
        assert 'Author: test_author' in output_file.read_text()

    # Test with invalid script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_content = """#!/bin/bash
exit 1
"""
        script_path.write_text(script_content, encoding='utf-8')

        # Verify that FailedHookException is raised
        with pytest.raises(FailedHookException):
            run_script_with_context(script_path, tmpdir, context)


# LLM-generated content at query #96
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mkstemp(suffix='.py')[1])
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Cleanup
    os.unlink(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #97
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'

    # Test case 2: Pre_prompt hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/fake/tmp_repo'), \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/tmp_repo'
        mock_work_in.assert_called()
        mock_run_script.assert_called_once_with('/fake/script.py', '/fake/tmp_repo')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='/fake/tmp_repo'), \
         patch('cookiecutter.hooks.work_in'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('/fake/repo_dir')


# LLM-generated content at query #98
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a simple hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

            context = {'project_name': 'test'}

            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )

            # Verify hook was executed
            assert os.path.exists(hook_script)

    # Test hook failure with project cleanup
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}

            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )

            # Verify project was cleaned up
            assert not os.path.exists(project_dir)

    # Test hook failure without project cleanup
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}

            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )

            # Verify project was NOT cleaned up
            assert os.path.exists(project_dir)

    # Test with undefined variable in hook
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a hook with undefined variable
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('{{ undefined_var }}')

            context = {'project_name': 'test'}

            with pytest.raises(UndefinedError):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )

            # Verify project was cleaned up
            assert not os.path.exists(project_dir)


# LLM-generated content at query #99
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')

        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')

        run_script(script_path, cwd=tmpdir)

    # Test failed execution of a script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')

        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test execution of an empty script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('')

        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test execution of a non-existent script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'non_existent_script.py')

        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)


# LLM-generated content at query #100
#--------------------------

```python
def test_find_hook():
    # Test case 1: Hook directory does not exist
    assert find_hook('pre_prompt', 'nonexistent_dir') is None

    # Test case 2: Hook directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_prompt', hooks_dir) is None

    # Test case 3: Hook directory exists with matching hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        Path(hook_file).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == hook_file

    # Test case 4: Hook directory exists with multiple matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_prompt.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert hook_file1 in result
        assert hook_file2 in result

    # Test case 5: Hook directory exists with invalid hooks (backup files)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py~')
        Path(hook_file).touch()
        assert find_hook('pre_prompt', hooks_dir) is None

    # Test case 6: Hook directory exists with non-matching hook names
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()
        assert find_hook('pre_prompt', hooks_dir) is None


# LLM-generated content at query #101
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Test case 1: No hook found
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)
    logger_spy = mocker.spy(logger, 'debug')
    run_hook('pre_gen_project', tmp_path, {})
    logger_spy.assert_called_with('No %s hook found', 'pre_gen_project')

    # Test case 2: Hook found and executed
    hook_script = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir()
    hook_script.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script)])
    run_script_spy = mocker.spy(hooks, 'run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    run_script_spy.assert_called_once_with(str(hook_script), tmp_path, {'project_name': 'test'})

    # Test case 3: Multiple hooks found and executed
    hook_script1 = tmp_path / 'hooks' / 'pre_gen_project.py'
    hook_script2 = tmp_path / 'hooks' / 'pre_gen_project.sh'
    hook_script1.parent.mkdir()
    hook_script1.write_text('print("Hook 1 executed")')
    hook_script2.write_text('echo "Hook 2 executed"')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(hook_script1), str(hook_script2)])
    run_script_spy = mocker.spy(hooks, 'run_script_with_context')
    run_hook('pre_gen_project', tmp_path, {'project_name': 'test'})
    assert run_script_spy.call_count == 2


# LLM-generated content at query #102
#--------------------------

```python
def test_run_hook():
    # Test case 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            run_hook('test_hook', '/fake/project_dir', {'key': 'value'})
            mock_find_hook.assert_called_once_with('test_hook')
            mock_logger.debug.assert_called_once_with('No %s hook found', 'test_hook')

    # Test case 2: Hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook_script.sh']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook('test_hook', '/fake/project_dir', {'key': 'value'})
                mock_find_hook.assert_called_once_with('test_hook')
                mock_run_script.assert_called_once_with('/fake/hook_script.sh', '/fake/project_dir', {'key': 'value'})
                mock_logger.debug.assert_called_once_with('Running hook %s', 'test_hook')

    # Test case 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/fake/hook_script1.sh', '/fake/hook_script2.sh']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook('test_hook', '/fake/project_dir', {'key': 'value'})
                mock_find_hook.assert_called_once_with('test_hook')
                mock_run_script.assert_has_calls([
                    call('/fake/hook_script1.sh', '/fake/project_dir', {'key': 'value'}),
                    call('/fake/hook_script2.sh', '/fake/project_dir', {'key': 'value'})
                ])
                mock_logger.debug.assert_called_once_with('Running hook %s', 'test_hook')


# LLM-generated content at query #103
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Mock the find_hook function to return a script path
    mocker.patch('cookiecutter.hooks.find_hook', return_value=['/path/to/script.py'])

    # Mock the run_script_with_context function
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Create a temporary directory for the project
    project_dir = tmp_path / 'project'
    project_dir.mkdir()

    # Create a sample context
    context = {'project_name': 'test_project'}

    # Call the function
    run_hook('pre_gen_project', project_dir, context)

    # Assert that run_script_with_context was called with the correct arguments
    mock_run_script.assert_called_once_with('/path/to/script.py', project_dir, context)

def test_run_hook_no_script(mocker, tmp_path, caplog):
    # Mock the find_hook function to return None (no script found)
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Mock the logger
    mocker.patch('cookiecutter.hooks.logger')

    # Create a temporary directory for the project
    project_dir = tmp_path / 'project'
    project_dir.mkdir()

    # Create a sample context
    context = {'project_name': 'test_project'}

    # Call the function
    run_hook('pre_gen_project', project_dir, context)

    # Assert that the logger was called with the correct message
    logger.debug.assert_called_once_with('No %s hook found', 'pre_gen_project')

def test_run_hook_multiple_scripts(mocker, tmp_path):
    # Mock the find_hook function to return multiple script paths
    mocker.patch('cookiecutter.hooks.find_hook', return_value=['/path/to/script1.py', '/path/to/script2.py'])

    # Mock the run_script_with_context function
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Create a temporary directory for the project
    project_dir = tmp_path / 'project'
    project_dir.mkdir()

    # Create a sample context
    context = {'project_name': 'test_project'}

    # Call the function
    run_hook('pre_gen_project', project_dir, context)

    # Assert that run_script_with_context was called for each script
    assert mock_run_script.call_count == 2
    mock_run_script.assert_any_call('/path/to/script1.py', project_dir, context)
    mock_run_script.assert_any_call('/path/to/script2.py', project_dir, context)


# LLM-generated content at query #104
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')
        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')
        run_script(script_path, cwd=tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test failure of a script with no shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)


# LLM-generated content at query #105
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    script_path = Path(tempfile.mkstemp(suffix='.py')[1])
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()
    context = {'name': 'World'}

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Cleanup
    os.unlink(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #106
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
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp_repo_dir.return_value = '/fake/tmp/repo'
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == '/fake/tmp/repo'
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp_repo_dir.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/fake/tmp/repo')

    # Test case 3: Pre_prompt hook found but execution fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp_repo_dir.return_value = '/fake/tmp/repo'
        mock_run_script.side_effect = FailedHookException('Script failed')
        repo_dir = Path('/fake/repo')
        with pytest.raises(FailedHookException) as excinfo:
            run_pre_prompt_hook(repo_dir)
        assert 'Pre-Prompt Hook script failed' in str(excinfo.value)
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp_repo_dir.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', '/fake/tmp/repo')


# LLM-generated content at query #107
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, World!")')
        run_script(str(script_path), cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('#!/bin/sh\necho "Hello, World!"')
        run_script(str(script_path), cwd=tmpdir)

    # Test failed execution of a script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('import sys\nsys.exit(1)')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), cwd=tmpdir)

    # Test execution of a non-existent script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'non_existent_script.py'
        with pytest.raises(FailedHookException):
            run_script(str(script_path), cwd=tmpdir)

    # Test execution of an empty script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'empty_script.py'
        script_path.write_text('')
        with pytest.raises(FailedHookException):
            run_script(str(script_path), cwd=tmpdir)


# LLM-generated content at query #108
#--------------------------

```python
def test_run_hook_from_repo_dir():
    # Test successful hook execution
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a valid hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('print("Hook executed")')

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
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
            assert not os.path.exists(project_dir)

    # Test hook failure without project deletion
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            # Create a failing hook script
            hook_dir = os.path.join(repo_dir, 'hooks')
            os.makedirs(hook_dir)
            hook_script = os.path.join(hook_dir, 'post_gen_project.py')
            with open(hook_script, 'w') as f:
                f.write('import sys; sys.exit(1)')

            context = {'project_name': 'test'}
            with pytest.raises(FailedHookException):
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name='post_gen_project',
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
            assert os.path.exists(project_dir)

    # Test no hook found
    with tempfile.TemporaryDirectory() as repo_dir:
        with tempfile.TemporaryDirectory() as project_dir:
            context = {'project_name': 'test'}
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name='post_gen_project',
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )


# LLM-generated content at query #109
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #110
#--------------------------

```python
def test_find_hook():
    # Test with non-existent hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test with empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_hook('pre_gen_project', tmpdir) is None

    # Test with valid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(valid_hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook_file]

    # Test with invalid hook file (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        invalid_hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(invalid_hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(backup_hook_file).touch()
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with multiple valid hook files
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


# LLM-generated content at query #111
#--------------------------

```python
def test_run_hook():
    # Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / 'project'
        project_dir.mkdir()
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()

        # Create a valid hook script
        hook_script = hooks_dir / 'post_gen_project.py'
        hook_script.write_text('print("Hook executed")')

        context = {'project_name': 'test_project'}

        # Test
        with work_in(project_dir):
            run_hook('post_gen_project', project_dir, context)

        # Assert
        assert hook_script.exists()


# LLM-generated content at query #112
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
        mock_find_hook.assert_called_once_with('pre_prompt')

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/tmp/fake_repo')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)

        assert result == Path('/tmp/fake_repo')
        mock_find_hook.assert_called_with('pre_prompt')
        mock_create_tmp.assert_called_once_with(repo_dir)
        mock_run_script.assert_called_once_with('/fake/repo/hooks/pre_prompt.py', str(mock_create_tmp.return_value))

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:

        mock_find_hook.return_value = ['/fake/repo/hooks/pre_prompt.py']
        mock_create_tmp.return_value = Path('/tmp/fake_repo')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None
        mock_run_script.side_effect = FailedHookException('Test error')

        repo_dir = Path('/fake/repo')
        with pytest.raises(FailedHookException) as exc_info:
            run_pre_prompt_hook(repo_dir)

        assert str(exc_info.value) == 'Pre-Prompt Hook script failed'
        assert exc_info.value.__cause__ is not None


# LLM-generated content at query #113
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    context = {'name': 'World'}
    script_path = Path(tempfile.mktemp(suffix='.py'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()

    # Execute
    run_script_with_context(script_path, cwd, context)

    # Cleanup
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #114
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
    os.unlink(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #115
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #116
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test when no pre_prompt hook is found
    with patch('cookiecutter.hooks.find_hook', return_value=None):
        result = run_pre_prompt_hook('dummy_repo_dir')
        assert result == 'dummy_repo_dir'

    # Test when pre_prompt hook is found and executed successfully
    with patch('cookiecutter.hooks.find_hook', return_value=['dummy_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        result = run_pre_prompt_hook('dummy_repo_dir')
        assert result == 'temp_repo_dir'
        mock_run_script.assert_called_once_with('dummy_script.py', 'temp_repo_dir')

    # Test when pre_prompt hook fails
    with patch('cookiecutter.hooks.find_hook', return_value=['dummy_script.py']), \
         patch('cookiecutter.hooks.create_tmp_repo_dir', return_value='temp_repo_dir'), \
         patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
        with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
            run_pre_prompt_hook('dummy_repo_dir')


# LLM-generated content at query #117
#--------------------------

```python
def test_run_script_with_context():
    # Setup
    script_content = "print('Hello, {{ name }}!')"
    context = {'name': 'World'}
    extension = '.py'
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=extension) as temp:
        temp.write(script_content)
        temp_path = temp.name

    # Execute
    run_script_with_context(temp_path, '.', context)

    # Teardown
    os.unlink(temp_path)


# LLM-generated content at query #118
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
    os.remove(script_path)
    os.rmdir(cwd)


# LLM-generated content at query #119
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

    # Test failure of a script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys; sys.exit(1)')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)

    # Test non-existent script
    with pytest.raises(FailedHookException):
        run_script('/nonexistent/script.py')

    # Test script without shebang
    with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
        f.write('echo "Hello, World!"')
        f.flush()
        script_path = f.name

    try:
        with pytest.raises(FailedHookException):
            run_script(script_path)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #120
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
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp_repo_dir.return_value = '/fake/tmp_repo_dir'
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/tmp_repo_dir'
        mock_run_script.assert_called_once_with('/fake/script.py', '/fake/tmp_repo_dir')

    # Test case 3: Pre_prompt hook found but execution failed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create_tmp_repo_dir, \
         patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_script') as mock_run_script:
        mock_find_hook.side_effect = [['/fake/script.py'], ['/fake/script.py']]
        mock_create_tmp_repo_dir.return_value = '/fake/tmp_repo_dir'
        mock_run_script.side_effect = FailedHookException('Test error')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook('/fake/repo_dir')


# LLM-generated content at query #121
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

    # Test finding no hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('non_existent_hook')
            assert result is None
        finally:
            os.chdir(original_cwd)

    # Test with no hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)

    # Test with invalid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file
        hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        Path(hook_file).touch()

        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)

    # Test with backup file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()

        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            result = find_hook('pre_gen_project')
            assert result is None
        finally:
            os.chdir(original_cwd)


# LLM-generated content at query #122
#--------------------------

```python
def test_find_hook():
    # Test finding a valid hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a valid hook file
        valid_hook = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [valid_hook]

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

        # Create a backup hook file
        backup_hook = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test finding invalid hook (wrong name)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create an invalid hook file
        invalid_hook = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(invalid_hook, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("test")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test with non-existent hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #123
#--------------------------

```python
def test_run_script_with_context():
    # Test with a simple Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, {{ name }}!")')

        context = {'name': 'World'}

        run_script_with_context(script_path, tmpdir, context)

    # Test with a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.sh'
        script_path.write_text('echo "Hello, {{ name }}!"')

        context = {'name': 'World'}

        run_script_with_context(script_path, tmpdir, context)

    # Test with a script that fails
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('import sys; sys.exit(1)')

        context = {}

        with pytest.raises(FailedHookException):
            run_script_with_context(script_path, tmpdir, context)

    # Test with a script that has undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('print("Hello, {{ undefined_var }}!")')

        context = {}

        with pytest.raises(UndefinedError):
            run_script_with_context(script_path, tmpdir, context)


# LLM-generated content at query #124
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
        non_existent_hooks_dir = os.path.join(tmpdir, 'non_existent_hooks_dir')
        result = find_hook('pre_gen_project', non_existent_hooks_dir)
        assert result is None

    # Test no matching hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a non-matching hook file
        non_matching_hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(non_matching_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Invalid hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test backup file is ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create a backup hook file
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook_file, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Backup hook")')

        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

    # Test multiple valid hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)

        # Create multiple valid hook files
        valid_hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook_file1, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Valid hook 1")')

        valid_hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(valid_hook_file2, 'w') as f:
            f.write('#!/bin/sh\necho "Valid hook 2"')

        result = find_hook('pre_gen_project', hooks_dir)
        assert len(result) == 2
        assert valid_hook_file1 in result
        assert valid_hook_file2 in result


# LLM-generated content at query #125
#--------------------------

```python
def test_run_script():
    # Test successful execution of a Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')
        run_script(script_path, cwd=tmpdir)

    # Test successful execution of a shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "Hello, World!"')
        run_script(script_path, cwd=tmpdir)

    # Test failure of a script with non-zero exit status
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import sys\nsys.exit(1)')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test failure of a script with no shebang
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('echo "Hello, World!"')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)

    # Test failure of a script with OSError
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('import os\nos.remove("nonexistent_file")')
        with pytest.raises(FailedHookException):
            run_script(script_path, cwd=tmpdir)


# LLM-generated content at query #126
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test'}

    # Mock find_hook to return a script path
    script_path = tmp_path / 'hooks' / f'{hook_name}.py'
    script_path.parent.mkdir()
    script_path.write_text('print("Hook executed")')
    mocker.patch('cookiecutter.hooks.find_hook', return_value=[str(script_path)])

    # Mock run_script_with_context
    mocker.patch('cookiecutter.hooks.run_script_with_context')

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    run_script_with_context.assert_called_once_with(str(script_path), project_dir, context)

def test_run_hook_no_hook_found(mocker, tmp_path, caplog):
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test'}

    # Mock find_hook to return None
    mocker.patch('cookiecutter.hooks.find_hook', return_value=None)

    # Execute
    run_hook(hook_name, project_dir, context)

    # Assert
    assert f'No {hook_name} hook found' in caplog.text


