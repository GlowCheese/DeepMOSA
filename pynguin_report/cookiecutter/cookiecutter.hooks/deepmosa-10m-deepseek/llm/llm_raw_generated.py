####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/some/path/hook_name.py", "hook_name")
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook("/some/path/hook_name.py~", "hook_name")
    assert result == False

def test_valid_hook_matching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown_hook.py", "unknown_hook")
    assert result == False

def test_valid_hook_non_matching_supported_not_backup():
    result = valid_hook("/some/path/pre_commit.py", "other_hook")
    assert result == False

def test_valid_hook_non_matching_supported_backup():
    result = valid_hook("/some/path/pre_commit.py~", "other_hook")
    assert result == False

def test_valid_hook_non_matching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown.py", "hook_name")
    assert result == False

def test_valid_hook_matching_unsupported_backup():
    result = valid_hook("/some/path/unknown_hook.py~", "unknown_hook")
    assert result == False

def test_valid_hook_non_matching_unsupported_backup():
    result = valid_hook("/some/path/unknown.py~", "other_hook")
    assert result == False


# LLM-generated content at query #2
#--------------------------

def test_run_hook_no_hooks_found():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('cookiecutter.hooks.find_hook', return_value=None):
            run_hook('pre_gen_project', tmpdir, {})

def test_run_hook_single_hook_executed():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_script = os.path.join(tmpdir, 'hook.py')
        with open(mock_script, 'w') as f:
            f.write('')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script]):
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('pre_gen_project', tmpdir, {'key': 'value'})
                mock_run.assert_called_once_with(mock_script, tmpdir, {'key': 'value'})

def test_run_hook_multiple_hooks_executed():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_script1 = os.path.join(tmpdir, 'hook1.py')
        mock_script2 = os.path.join(tmpdir, 'hook2.py')
        with open(mock_script1, 'w') as f:
            f.write('')
        with open(mock_script2, 'w') as f:
            f.write('')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script1, mock_script2]):
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('pre_gen_project', tmpdir, {'key': 'value'})
                assert mock_run.call_count == 2
                mock_run.assert_any_call(mock_script1, tmpdir, {'key': 'value'})
                mock_run.assert_any_call(mock_script2, tmpdir, {'key': 'value'})

def test_run_hook_empty_context():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_script = os.path.join(tmpdir, 'hook.py')
        with open(mock_script, 'w') as f:
            f.write('')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script]):
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('post_gen_project', tmpdir, {})
                mock_run.assert_called_once_with(mock_script, tmpdir, {})

def test_run_hook_with_none_project_dir():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_script = os.path.join(tmpdir, 'hook.py')
        with open(mock_script, 'w') as f:
            f.write('')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script]):
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('pre_gen_project', None, {'key': 'value'})
                mock_run.assert_called_once_with(mock_script, None, {'key': 'value'})


# LLM-generated content at query #3
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/some/path/hook_name.py", "hook_name")
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook("/some/path/hook_name.py~", "hook_name")
    assert result == False

def test_valid_hook_matching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown_hook.py", "unknown_hook")
    assert result == False

def test_valid_hook_not_matching_supported_not_backup():
    result = valid_hook("/some/path/other_hook.py", "hook_name")
    assert result == False

def test_valid_hook_not_matching_supported_backup():
    result = valid_hook("/some/path/other_hook.py~", "hook_name")
    assert result == False

def test_valid_hook_not_matching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown.py", "hook_name")
    assert result == False

def test_valid_hook_matching_unsupported_backup():
    result = valid_hook("/some/path/unknown_hook.py~", "unknown_hook")
    assert result == False

def test_valid_hook_not_matching_unsupported_backup():
    result = valid_hook("/some/path/unknown.py~", "hook_name")
    assert result == False


# LLM-generated content at query #4
#--------------------------

def test_find_hook_with_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_with_empty_hooks_dir():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_with_valid_hook():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'pre_gen_project.py')
        open(hook_file, 'a').close()
        result = find_hook('pre_gen_project', tmpdir)
        assert result == [os.path.abspath(hook_file)]

def test_find_hook_with_backup_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'pre_gen_project.py~')
        open(hook_file, 'a').close()
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_with_unsupported_hook():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'unsupported_hook.py')
        open(hook_file, 'a').close()
        result = find_hook('unsupported_hook', tmpdir)
        assert result is None

def test_find_hook_with_mismatched_hook_name():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'post_gen_project.py')
        open(hook_file, 'a').close()
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file1 = os.path.join(tmpdir, 'pre_gen_project.py')
        hook_file2 = os.path.join(tmpdir, 'pre_gen_project.sh')
        open(hook_file1, 'a').close()
        open(hook_file2, 'a').close()
        result = find_hook('pre_gen_project', tmpdir)
        expected = sorted([os.path.abspath(hook_file1), os.path.abspath(hook_file2)])
        assert sorted(result) == expected


# LLM-generated content at query #5
#--------------------------

def test_run_pre_prompt_hook_with_no_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_with_empty_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_with_invalid_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_prompt.txt').write_text('test')
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_prompt~').write_text('test')
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_with_wrong_hook_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'post_gen_project.py').write_text('test')
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_with_valid_python_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)

def test_run_pre_prompt_hook_with_valid_shell_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/sh\nexit 0')
        os.chmod(script, 0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)

def test_run_pre_prompt_hook_with_failing_python_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_failing_shell_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/sh\nexit 1')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_empty_script_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_missing_shebang():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('exit 0')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_multiple_valid_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('import sys\nsys.exit(0)')
        script2 = hooks_dir / 'pre_prompt'
        script2.write_text('#!/bin/sh\nexit 0')
        os.chmod(script2, 0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)

def test_run_pre_prompt_hook_creates_tmp_copy():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(0)')
        test_file = Path(tmpdir) / 'test.txt'
        test_file.write_text('original')
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        copied_test_file = Path(result) / 'test.txt'
        assert copied_test_file.read_text() == 'original'


# LLM-generated content at query #6
#--------------------------

def test_hooks_dir_is_not_a_directory():
    os.path.isdir = lambda x: False
    result = find_hook('some_hook', 'hooks')
    assert result is None


# LLM-generated content at query #7
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook('/path/to/hook.py', 'hook')
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook('/path/to/hook.py~', 'hook')
    assert result == False

def test_valid_hook_matching_unsupported():
    result = valid_hook('/path/to/unknown.py', 'unknown')
    assert result == False

def test_valid_hook_nonmatching_supported():
    result = valid_hook('/path/to/other_hook.py', 'hook')
    assert result == False

def test_valid_hook_nonmatching_unsupported():
    result = valid_hook('/path/to/unknown.py', 'hook')
    assert result == False

def test_valid_hook_empty_hook_name():
    result = valid_hook('/path/to/.py', '')
    assert result == False

def test_valid_hook_file_without_extension():
    result = valid_hook('/path/to/hook', 'hook')
    assert result == True

def test_valid_hook_file_with_multiple_dots():
    result = valid_hook('/path/to/hook.test.py', 'hook.test')
    assert result == False


# LLM-generated content at query #8
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_no_hook_found():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: None
    hooks.find_hook = mock_find_hook
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_hook_fails_with_failed_hook_exception():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    mock_rmtree = lambda path: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass

def test_run_hook_from_repo_dir_hook_fails_with_undefined_error():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(jinja2.UndefinedError('Undefined variable'))
    mock_rmtree = lambda path: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except jinja2.UndefinedError:
        pass

def test_run_hook_from_repo_dir_hook_fails_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = False
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    mock_rmtree = lambda path: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path("/some/test/dir")
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #10
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts_returns_original_repo_dir():
    repo_dir = '/some/path'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #11
#--------------------------

def test_run_hook_from_repo_dir_delete_on_failure():
    mock_repo_dir = "/fake/repo"
    mock_project_dir = "/fake/project"
    mock_context = {}
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in, \
         unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(mock_repo_dir, "pre_gen_project", mock_project_dir, mock_context, True)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_rmtree.assert_called_once_with(mock_project_dir)
        mock_logger.exception.assert_called_once()


# LLM-generated content at query #12
#--------------------------

def test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir():
    original_repo_dir = "/some/path"
    result = run_pre_prompt_hook(original_repo_dir)
    assert result == original_repo_dir


# LLM-generated content at query #13
#--------------------------

def test_run_hook_no_hooks_dir():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {}
        with patch('cookiecutter.hooks.find_hook', return_value=None) as mock_find:
            run_hook('pre_gen_project', tmpdir, context)
            mock_find.assert_called_once_with('pre_gen_project')
    pass


def test_run_hook_with_scripts():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {}
        mock_script = os.path.join(tmpdir, 'script.py')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script]) as mock_find:
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('post_gen_project', tmpdir, context)
                mock_find.assert_called_once_with('post_gen_project')
                mock_run.assert_called_once_with(mock_script, tmpdir, context)
    pass


def test_run_hook_multiple_scripts():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {}
        mock_script1 = os.path.join(tmpdir, 'script1.py')
        mock_script2 = os.path.join(tmpdir, 'script2.py')
        with patch('cookiecutter.hooks.find_hook', return_value=[mock_script1, mock_script2]) as mock_find:
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook('pre_gen_project', tmpdir, context)
                mock_find.assert_called_once_with('pre_gen_project')
                assert mock_run.call_count == 2
                mock_run.assert_any_call(mock_script1, tmpdir, context)
                mock_run.assert_any_call(mock_script2, tmpdir, context)
    pass


# LLM-generated content at query #14
#--------------------------

def test_run_script_success_python():
    import tempfile
    import os
    script_content = "print('hello')"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_success_non_python():
    import tempfile
    import os
    script_content = "#!/bin/sh\necho hello"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_failure_exit_status():
    import tempfile
    import os
    script_content = "import sys; sys.exit(1)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
        assert False
    except FailedHookException as e:
        assert "exit status: 1" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_failure_enoexec():
    import tempfile
    import os
    script_content = ""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
        assert False
    except FailedHookException as e:
        assert "empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_failure_oserror():
    import tempfile
    import os
    script_path = "/non/existent/path/script.sh"
    try:
        run_script(script_path)
        assert False
    except FailedHookException as e:
        assert "Hook script failed" in str(e)

def test_run_script_with_cwd():
    import tempfile
    import os
    script_content = "import os; print(os.getcwd())"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    cwd = tempfile.mkdtemp()
    try:
        run_script(script_path, cwd=cwd)
    finally:
        os.unlink(script_path)
        os.rmdir(cwd)


# LLM-generated content at query #15
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/path/script.py"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp/file.py"
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env.from_string.return_value = mock_template
    with patch("tempfile.NamedTemporaryFile") as mock_named_temp, patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("builtins.open", mock_open(read_data="template content")), patch("pathlib.Path.read_text") as mock_read_text:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_create_env.return_value = mock_env
        mock_read_text.return_value = "template content"
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_create_env.assert_called_once_with(context)
        mock_env.from_string.assert_called_once_with("template content")
        mock_template.render.assert_called_once_with(**context)
        mock_temp_file.write.assert_called_once_with(b"rendered content")
        mock_run_cript.assert_called_once_with("/fake/temp/file.py", cwd)

def test_run_script_with_context_passes_correct_extension_to_temp_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/path/script.sh"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp/file.sh"
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env.from_string.return_value = mock_template
    with patch("tempfile.NamedTemporaryFile") as mock_named_temp, patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("builtins.open", mock_open(read_data="template content")), patch("pathlib.Path.read_text") as mock_read_text:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_create_env.return_value = mock_env
        mock_read_text.return_value = "template content"
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_named_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')

def test_run_script_with_context_handles_empty_extension():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/path/script"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp/file"
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env.from_string.return_value = mock_template
    with patch("tempfile.NamedTemporaryFile") as mock_named_temp, patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("builtins.open", mock_open(read_data="template content")), patch("pathlib.Path.read_text") as mock_read_text:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_create_env.return_value = mock_env
        mock_read_text.return_value = "template content"
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_named_temp.assert_called_once_with(delete=False, mode='wb', suffix='')

def test_run_script_with_context_uses_utf8_encoding():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/path/script.py"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp/file.py"
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content with unicode: é"
    mock_env.from_string.return_value = mock_template
    with patch("tempfile.NamedTemporaryFile") as mock_named_temp, patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("builtins.open", mock_open(read_data="template content")), patch("pathlib.Path.read_text") as mock_read_text:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_create_env.return_value = mock_env
        mock_read_text.return_value = "template content"
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_temp_file.write.assert_called_once_with("rendered content with unicode: é".encode('utf-8'))

def test_run_script_with_context_passes_context_to_template_render():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/path/script.py"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject", "version": "1.0"}, "extra_key": "value"}
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp/file.py"
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env.from_string.return_value = mock_template
    with patch("tempfile.NamedTemporaryFile") as mock_named_temp, patch("cookiecutter.hooks.create_env_with_context") as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("builtins.open", mock_open(read_data="template content")), patch("pathlib.Path.read_text") as mock_read_text:
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_create_env.return_value = mock_env
        mock_read_text.return_value = "template content"
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_template.render.assert_called_once_with(**context)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    test_script_path = '/test/script.py.j2'
    test_cwd = '/test/cwd'
    test_context = {'cookiecutter': {'project_name': 'TestProject'}}
    test_contents = 'print("{{ cookiecutter.project_name }}")'
    expected_output = 'print("TestProject")'
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = expected_output
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, patch('cookiecutter.hooks.Path') as mock_path_class, patch('cookiecutter.hooks.run_script') as mock_run_script, patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = test_contents
        mock_path_class.return_value = mock_path_instance
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp123.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(test_script_path, test_cwd, test_context)
        mock_create_env.assert_called_once_with(test_context)
        mock_env.from_string.assert_called_once_with(test_contents)
        mock_template.render.assert_called_once_with(**test_context)
        mock_temp.write.assert_called_once_with(expected_output.encode('utf-8'))
        mock_run_script.assert_called_once_with('/tmp/temp123.py', test_cwd)

def test_run_script_with_context_preserves_file_extension():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    test_script_path = '/test/script.sh.j2'
    test_cwd = '/test/cwd'
    test_context = {'cookiecutter': {'key': 'value'}}
    test_contents = 'echo "{{ cookiecutter.key }}"'
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'echo "value"'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, patch('cookiecutter.hooks.Path') as mock_path_class, patch('cookiecutter.hooks.run_script') as mock_run_script, patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = test_contents
        mock_path_class.return_value = mock_path_instance
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp123.sh'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(test_script_path, test_cwd, test_context)
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.j2')

def test_run_script_with_context_handles_path_object():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    test_script_path = Path('/test/script.py.j2')
    test_cwd = Path('/test/cwd')
    test_context = {'cookiecutter': {'name': 'Test'}}
    test_contents = 'print("{{ cookiecutter.name }}")'
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'print("Test")'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, patch('cookiecutter.hooks.Path') as mock_path_class, patch('cookiecutter.hooks.run_script') as mock_run_script, patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = test_contents
        mock_path_class.return_value = mock_path_instance
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp123.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(test_script_path, test_cwd, test_context)
        mock_path_class.assert_called_once_with(test_script_path)
        mock_run_script.assert_called_once_with('/tmp/temp123.py', test_cwd)

def test_run_script_with_context_passes_context_to_environment():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    test_script_path = '/test/script.py.j2'
    test_cwd = '/test/cwd'
    test_context = {'cookiecutter': {'_jinja2_env_vars': {'autoescape': True}, 'var': 'test'}}
    test_contents = 'template content'
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered content'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, patch('cookiecutter.hooks.Path') as mock_path_class, patch('cookiecutter.hooks.run_script') as mock_run_script, patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = test_contents
        mock_path_class.return_value = mock_path_instance
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp123.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(test_script_path, test_cwd, test_context)
        mock_create_env.assert_called_once_with(test_context)
        mock_template.render.assert_called_once_with(**test_context)


# LLM-generated content at query #17
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    valid_hook_path = os.path.join(hooks_subdir, 'pre_gen_project.py')
    open(valid_hook_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(valid_hook_path)]

def test_find_hook_with_no_hooks_directory():
    import tempfile, os
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    os.rmdir(temp_dir)
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    backup_hook_path = os.path.join(hooks_subdir, 'pre_gen_project.py~')
    open(backup_hook_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_unsupported_hook_name():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    unsupported_hook_path = os.path.join(hooks_subdir, 'unsupported_hook.py')
    open(unsupported_hook_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('unsupported_hook', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_mismatched_hook_name():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    other_hook_path = os.path.join(hooks_subdir, 'post_gen_project.py')
    open(other_hook_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_subdir = os.path.join(temp_dir, 'hooks')
    os.mkdir(hooks_subdir)
    hook1_path = os.path.join(hooks_subdir, 'pre_gen_project.py')
    hook2_path = os.path.join(hooks_subdir, 'pre_gen_project.sh')
    open(hook1_path, 'w').close()
    open(hook2_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    expected = sorted([os.path.abspath(hook1_path), os.path.abspath(hook2_path)])
    assert sorted(result) == expected

def test_find_hook_with_custom_hooks_dir():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    custom_hooks_dir = os.path.join(temp_dir, 'custom_hooks')
    os.mkdir(custom_hooks_dir)
    valid_hook_path = os.path.join(custom_hooks_dir, 'pre_gen_project.py')
    open(valid_hook_path, 'w').close()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    result = find_hook('pre_gen_project', 'custom_hooks')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(valid_hook_path)]


# LLM-generated content at query #18
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import create_env_with_context
    from cookiecutter.hooks import run_script_with_context
    
    test_script_path = "/tmp/test_script.py"
    test_cwd = "/tmp"
    test_context = {"cookiecutter": {"project_name": "test"}}
    
    Path(test_script_path).write_text("print('{{ cookiecutter.project_name }}')", encoding='utf-8')
    
    run_script_with_context(test_script_path, test_cwd, test_context)
    
    temp_files = list(Path(tempfile.gettempdir()).glob("*.py"))
    assert any(temp_file.suffix == ".py" for temp_file in temp_files)
    
    os.remove(test_script_path)
    for temp_file in temp_files:
        if temp_file.name.startswith("tmp"):
            os.remove(temp_file)


# LLM-generated content at query #19
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py']), patch('os.path.isdir', return_value=True), patch('valid_hook', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_no_hooks_directory():
    import os
    from unittest.mock import patch
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=[]), patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_no_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=['invalid.py', 'backup.py~']), patch('os.path.isdir', return_value=True), patch('valid_hook', return_value=False):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file1, 'w') as f1, open(hook_file2, 'w') as f2:
            f1.write(''); f2.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py', 'post_gen_project.py']), patch('os.path.isdir', return_value=True), patch('valid_hook', side_effect=lambda f, h: f in ['pre_gen_project.py', 'post_gen_project.py']):
            result = find_hook('pre_gen_project', hooks_dir)
    expected = [os.path.abspath(hook_file1), os.path.abspath(hook_file2)]
    assert sorted(result) == sorted(expected)

def test_find_hook_uses_absolute_paths():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py']), patch('os.path.isdir', return_value=True), patch('valid_hook', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert os.path.isabs(result[0])


# LLM-generated content at query #20
#--------------------------

def test_run_hook_no_hooks_found():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_valid_hook():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_gen_project.py'
        hook_file.write_text('print("Hello")')
        context = {}
        run_hook('pre_gen_project', tmpdir, context)


def test_run_hook_with_jinja_context():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_gen_project.py'
        hook_file.write_text('{{ cookiecutter.project_name }}')
        context = {'cookiecutter': {'project_name': 'TestProject'}}
        run_hook('pre_gen_project', tmpdir, context)


def test_run_hook_multiple_scripts():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file1 = hooks_dir / 'pre_gen_project.py'
        hook_file1.write_text('print("First")')
        hook_file2 = hooks_dir / 'pre_gen_project.sh'
        hook_file2.write_text('#!/bin/bash\necho "Second"')
        os.chmod(hook_file2, 0o755)
        context = {}
        run_hook('pre_gen_project', tmpdir, context)


def test_run_hook_invalid_hook_ignored():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        valid_hook = hooks_dir / 'pre_gen_project.py'
        valid_hook.write_text('print("Valid")')
        invalid_hook = hooks_dir / 'invalid_hook.py'
        invalid_hook.write_text('print("Invalid")')
        context = {}
        run_hook('pre_gen_project', tmpdir, context)


def test_run_hook_backup_file_ignored():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        backup_hook = hooks_dir / 'pre_gen_project.py~'
        backup_hook.write_text('print("Backup")')
        context = {}
        run_hook('pre_gen_project', tmpdir, context)


# LLM-generated content at query #21
#--------------------------

def test_no_hook_found():
    from cookiecutter.hooks import run_hook
    from pathlib import Path
    import logging
    logger = logging.getLogger(__name__)
    scripts = []
    result = run_hook('pre_gen_project', Path('/tmp'), {})
    assert result is None


# LLM-generated content at query #22
#--------------------------

def test_run_pre_prompt_hook_with_no_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_empty_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_non_matching_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'post_gen_project.py').touch()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_prompt.py~').touch()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_python_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.name == repo_dir.name

def test_run_pre_prompt_hook_with_valid_shell_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/sh\nexit 0')
        os.chmod(script, 0o755)
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.name == repo_dir.name

def test_run_pre_prompt_hook_with_failing_python_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_failing_shell_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/sh\nexit 1')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_empty_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_missing_shebang():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('exit 0')
        os.chmod(script, 0o755)
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_with_multiple_valid_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('import sys\nsys.exit(0)')
        script2 = hooks_dir / 'pre_prompt'
        script2.write_text('#!/bin/sh\nexit 0')
        os.chmod(script2, 0o755)
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.name == repo_dir.name


# LLM-generated content at query #23
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('pre_gen_project', hooks_dir)
    expected = [hook_file_path]
    assert result == expected

def test_find_hook_with_no_hooks_directory():
    import os
    from unittest.mock import patch
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=[]):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py~']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook_name():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['unsupported_hook.py']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('unsupported_hook', hooks_dir)
    assert result is None

def test_find_hook_with_mismatched_hook_name():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['post_gen_project.py']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_files():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file_path2 = os.path.join(hooks_dir, 'pre_gen_project.txt')
        with open(hook_file_path1, 'w') as f:
            f.write('')
        with open(hook_file_path2, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py', 'pre_gen_project.txt']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    result = find_hook('pre_gen_project', hooks_dir)
    expected = [hook_file_path1]
    assert result == expected


# LLM-generated content at query #24
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'pre_gen_project.sh'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
                assert result == [os.path.abspath(os.path.join(hooks_dir, hook_file))]

def test_find_hook_with_no_hooks_dir():
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        with patch('os.path.isdir', return_value=False):
            result = find_hook('any_hook', hooks_dir)
            assert result is None

def test_find_hook_with_empty_hooks_dir():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=[]):
            result = find_hook('any_hook', hooks_dir)
            assert result is None

def test_find_hook_with_no_valid_hook():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'some_file.txt'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=False):
                result = find_hook('pre_gen_project', hooks_dir)
                assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_files = ['pre_gen_project.sh', 'post_gen_project.sh']
        for hf in hook_files:
            with open(os.path.join(hooks_dir, hf), 'w') as f:
                f.write('')
        with patch('os.listdir', return_value=hook_files):
            with patch('valid_hook', return_value=True):
                result = find_hook('any_hook', hooks_dir)
                expected = [os.path.abspath(os.path.join(hooks_dir, hf)) for hf in hook_files]
                assert result == expected


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_when_configured():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = True
    exception = FailedHookException("Hook failed")
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
        with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                    mock_run_hook.side_effect = exception
                    try:
                        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                    except FailedHookException:
                        pass
                    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #26
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'pre_gen_project.py'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
                assert result == [os.path.abspath(os.path.join(hooks_dir, hook_file))]

def test_find_hook_with_no_hooks_directory():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_empty_hooks_directory():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=[]):
            result = find_hook('pre_gen_project', hooks_dir)
            assert result is None

def test_find_hook_with_no_matching_hook():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'post_gen_project.py'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=False):
                result = find_hook('pre_gen_project', hooks_dir)
                assert result is None

def test_find_hook_with_multiple_matching_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_files = ['pre_gen_project.py', 'pre_gen_project.sh']
        for hf in hook_files:
            with open(os.path.join(hooks_dir, hf), 'w') as f:
                f.write('')
        with patch('os.listdir', return_value=hook_files):
            with patch('valid_hook', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
                expected = [os.path.abspath(os.path.join(hooks_dir, hf)) for hf in hook_files]
                assert sorted(result) == sorted(expected)


# LLM-generated content at query #27
#--------------------------

def test_find_hook_with_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_with_empty_hooks_dir(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_valid_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project.py')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file)]

def test_find_hook_with_unsupported_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('unsupported_hook.py')
    hook_file.write('')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

def test_find_hook_with_backup_file(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project.py~')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_mismatched_name(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('post_gen_project.py')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_multiple_valid_hooks(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file1 = hooks_dir.join('pre_gen_project.py')
    hook_file1.write('')
    hook_file2 = hooks_dir.join('post_gen_project.py')
    hook_file2.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file1)]


# LLM-generated content at query #28
#--------------------------

def test_run_hook_from_repo_dir_does_not_delete_project_on_failure_when_flag_false():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {}
    delete_project_on_failure = False
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in, unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree, unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_rmtree.assert_not_called()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_run_pre_prompt_hook_no_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir


def test_run_pre_prompt_hook_empty_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir


def test_run_pre_prompt_hook_invalid_hook_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_prompt.txt').touch()
        (hooks_dir / 'pre_prompt~').touch()
        (hooks_dir / 'post_gen_project.py').touch()
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir


def test_run_pre_prompt_hook_valid_python_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)


def test_run_pre_prompt_hook_valid_shell_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/sh\nexit 0')
        script.chmod(0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)


def test_run_pre_prompt_hook_script_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True


def test_run_pre_prompt_hook_empty_script_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.touch()
        script.chmod(0o755)
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True


def test_run_pre_prompt_hook_multiple_valid_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('import sys\nsys.exit(0)')
        script2 = hooks_dir / 'pre_prompt'
        script2.write_text('#!/bin/sh\nexit 0')
        script2.chmod(0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert 'cookiecutter' in str(result)


# LLM-generated content at query #2
#--------------------------

def test_run_hook_no_hooks_found():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_valid_script():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_gen_project.py'
        script_path.write_text('print("Hello")')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_multiple_valid_scripts():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_gen_project.py'
        script1.write_text('print("Hello")')
        script2 = hooks_dir / 'pre_gen_project.sh'
        script2.write_text('#!/bin/bash\necho "World"')
        os.chmod(script2, 0o755)
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_ignores_backup_files():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        backup_script = hooks_dir / 'pre_gen_project.py~'
        backup_script.write_text('print("Backup")')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_ignores_unsupported_hook_names():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        unsupported_script = hooks_dir / 'unsupported_hook.py'
        unsupported_script.write_text('print("Unsupported")')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_jinja_context():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_gen_project.py'
        script_content = 'print("{{ greeting }}")'
        script_path.write_text(script_content)
        project_dir = Path(tmpdir)
        context = {'greeting': 'Hello World'}
        run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #3
#--------------------------

def test_run_hook_no_hooks_dir():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_empty_hooks_dir():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_no_matching_script():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'post_gen_project.py').write_text('')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_backup_file():
    import tempfile
    import os
    from pathlib: Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_gen_project.py~').write_text('')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_unsupported_hook_name():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'unsupported_hook.py').write_text('')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('unsupported_hook', project_dir, context)


def test_run_hook_with_valid_script():
    import tempfile
    import os
    import sys
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_content = "import sys\nsys.exit(0)"
        (hooks_dir / 'pre_gen_project.py').write_text(script_content)
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_jinja_template():
    import tempfile
    import os
    import sys
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_content = "{{ cookiecutter.project_name }}"
        (hooks_dir / 'pre_gen_project.py').write_text(script_content)
        project_dir = Path(tmpdir)
        context = {'cookiecutter': {'project_name': 'test_project'}}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_multiple_scripts():
    import tempfile
    import os
    import sys
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'pre_gen_project.py').write_text('import sys\nsys.exit(0)')
        (hooks_dir / 'pre_gen_project.sh').write_text('#!/bin/bash\nexit 0')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #4
#--------------------------

def test_find_hook_with_valid_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(test_hooks_dir, 'pre_gen_project.py')
    open(hook_file_path, 'a').close()
    with patch('os.listdir', return_value=['pre_gen_project.py']), patch('os.path.isdir', return_value=True), patch('os.path.abspath', side_effect=lambda x: x), patch('os.path.join', side_effect=lambda a, b: os.path.join(a, b)):
        result = find_hook('pre_gen_project', test_hooks_dir)
    assert result == [hook_file_path]

def test_find_hook_with_no_hooks_dir():
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_empty_hooks_dir():
    with patch('os.listdir', return_value=[]), patch('os.path.isdir', return_value=True):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(test_hooks_dir, 'pre_gen_project.py~')
    open(hook_file_path, 'a').close()
    with patch('os.listdir', return_value=['pre_gen_project.py~']), patch('os.path.isdir', return_value=True), patch('os.path.abspath', side_effect=lambda x: x), patch('os.path.join', side_effect=lambda a, b: os.path.join(a, b)):
        result = find_hook('pre_gen_project', test_hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(test_hooks_dir, 'unsupported_hook.py')
    open(hook_file_path, 'a').close()
    with patch('os.listdir', return_value=['unsupported_hook.py']), patch('os.path.isdir', return_value=True), patch('os.path.abspath', side_effect=lambda x: x), patch('os.path.join', side_effect=lambda a, b: os.path.join(a, b)):
        result = find_hook('unsupported_hook', test_hooks_dir)
    assert result is None

def test_find_hook_with_mismatched_hook_name():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(test_hooks_dir, 'post_gen_project.py')
    open(hook_file_path, 'a').close()
    with patch('os.listdir', return_value=['post_gen_project.py']), patch('os.path.isdir', return_value=True), patch('os.path.abspath', side_effect=lambda x: x), patch('os.path.join', side_effect=lambda a, b: os.path.join(a, b)):
        result = find_hook('pre_gen_project', test_hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path1 = os.path.join(test_hooks_dir, 'pre_gen_project.py')
    hook_file_path2 = os.path.join(test_hooks_dir, 'pre_gen_project.sh')
    open(hook_file_path1, 'a').close()
    open(hook_file_path2, 'a').close()
    with patch('os.listdir', return_value=['pre_gen_project.py', 'pre_gen_project.sh']), patch('os.path.isdir', return_value=True), patch('os.path.abspath', side_effect=lambda x: x), patch('os.path.join', side_effect=lambda a, b: os.path.join(a, b)):
        result = find_hook('pre_gen_project', test_hooks_dir)
    assert set(result) == {hook_file_path1, hook_file_path2}


# LLM-generated content at query #5
#--------------------------

def test_hooks_dir_is_not_directory():
    os.path.isdir = lambda x: False


# LLM-generated content at query #6
#--------------------------

def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('some_dir')
    with unittest.mock.patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = hooks.run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, mock_open, MagicMock
    script_path = '/fake/script.py.j2'
    cwd = '/fake/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_file = MagicMock()
    mock_file.name = '/fake/tempfile.py'
    with patch('tempfile.NamedTemporaryFile') as mock_temp, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text') as mock_read_text:
        mock_temp.return_value.__enter__.return_value = mock_file
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = 'rendered content'
        mock_read_text.return_value = 'template content'
        run_script_with_context(script_path, cwd, context)
        mock_read_text.assert_called_once_with(encoding='utf-8')
        mock_create_env.assert_called_once_with(context)
        mock_env.from_string.assert_called_once_with('template content')
        mock_template.render.assert_called_once_with(**context)
        mock_file.write.assert_called_once_with(b'rendered content')
        mock_run_script.assert_called_once_with('/fake/tempfile.py', cwd)


# LLM-generated content at query #8
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


def test_run_hook_from_repo_dir_no_hook_found():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': None
    hooks.find_hook = mock_find_hook
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


def test_run_hook_from_repo_dir_hook_fails_with_failed_hook_exception():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    mock_rmtree = lambda path: None
    utils.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass


def test_run_hook_from_repo_dir_hook_fails_with_undefined_error():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(jinja2.UndefinedError('Undefined variable'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    mock_rmtree = lambda path: None
    utils.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except jinja2.UndefinedError:
        pass


def test_run_hook_from_repo_dir_hook_fails_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'name': 'test'}}
    delete_project_on_failure = False
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    mock_rmtree = lambda path: None
    utils.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass


# LLM-generated content at query #9
#--------------------------

def test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir():
    from cookiecutter.hooks import run_pre_prompt_hook
    from pathlib import Path
    import tempfile
    import os
    test_dir = Path(tempfile.mkdtemp())
    result = run_pre_prompt_hook(test_dir)
    assert result == test_dir
    os.rmdir(test_dir)


# LLM-generated content at query #10
#--------------------------

def test_run_pre_prompt_hook_no_scripts():
    repo_dir = Path("/some/repo")
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #11
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/script.py.j2"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env = Mock()
    mock_env.from_string.return_value = mock_template
    with patch("cookiecutter.hooks.create_env_with_context", return_value=mock_env) as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("pathlib.Path.read_text", return_value="original content") as mock_read_text, patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_create_env.assert_called_once_with(context)
        mock_read_text.assert_called_once_with(encoding='utf-8')
        mock_env.from_string.assert_called_once_with("original content")
        mock_template.render.assert_called_once_with(**context)
        mock_temp.write.assert_called_once_with(b'rendered content')
        mock_run_cript.assert_called_once_with("/fake/temp.py", cwd)

def test_run_script_with_context_preserves_file_extension():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/script.sh.j2"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template = Mock()
    mock_template.render.return_value = "rendered bash script"
    mock_env = Mock()
    mock_env.from_string.return_value = mock_template
    with patch("cookiecutter.hooks.create_env_with_context", return_value=mock_env) as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("pathlib.Path.read_text", return_value="original bash") as mock_read_text, patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.sh"
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.j2')
        mock_run_cript.assert_called_once_with("/fake/temp.sh", cwd)

def test_run_script_with_context_handles_empty_context():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/script.py.j2"
    cwd = "/fake/cwd"
    context = {}
    mock_template = Mock()
    mock_template.render.return_value = "rendered with empty context"
    mock_env = Mock()
    mock_env.from_string.return_value = mock_template
    with patch("cookiecutter.hooks.create_env_with_context", return_value=mock_env) as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("pathlib.Path.read_text", return_value="original") as mock_read_text, patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_create_env.assert_called_once_with(context)
        mock_template.render.assert_called_once_with(**context)

def test_run_script_with_context_uses_utf8_encoding():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/script.py.j2"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"name": "Test"}}
    mock_template = Mock()
    mock_template.render.return_value = "rendered with special chars é"
    mock_env = Mock()
    mock_env.from_string.return_value = mock_template
    with patch("cookiecutter.hooks.create_env_with_context", return_value=mock_env) as mock_create_env, patch("cookiecutter.hooks.run_script") as mock_run_script, patch("pathlib.Path.read_text", return_value="original é") as mock_read_text, patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_read_text.assert_called_once_with(encoding='utf-8')
        mock_temp.write.assert_called_once_with(b'rendered with special chars \xc3\xa9')


# LLM-generated content at query #12
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(test_hooks_dir, 'pre_gen_project.py')
    open(hook_file_path, 'a').close()
    with patch('os.path.isdir', return_value=True), patch('os.listdir', return_value=['pre_gen_project.py']), patch('valid_hook', return_value=True):
        result = find_hook('pre_gen_project', test_hooks_dir)
    os.remove(hook_file_path)
    os.rmdir(test_hooks_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_no_hooks_directory():
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    with patch('os.path.isdir', return_value=True), patch('os.listdir', return_value=[]):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_no_valid_hooks():
    with patch('os.path.isdir', return_value=True), patch('os.listdir', return_value=['invalid.py', 'backup.py~']), patch('valid_hook', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    test_hooks_dir = tempfile.mkdtemp()
    hook_file1_path = os.path.join(test_hooks_dir, 'pre_gen_project.py')
    hook_file2_path = os.path.join(test_hooks_dir, 'post_gen_project.py')
    open(hook_file1_path, 'a').close()
    open(hook_file2_path, 'a').close()
    with patch('os.path.isdir', return_value=True), patch('os.listdir', return_value=['pre_gen_project.py', 'post_gen_project.py']), patch('valid_hook', side_effect=lambda f, n: f in ['pre_gen_project.py', 'post_gen_project.py']):
        result = find_hook('pre_gen_project', test_hooks_dir)
    os.remove(hook_file1_path)
    os.remove(hook_file2_path)
    os.rmdir(test_hooks_dir)
    assert result == [os.path.abspath(hook_file1_path), os.path.abspath(hook_file2_path)]

def test_find_hook_uses_default_hooks_dir():
    with patch('os.path.isdir', return_value=True), patch('os.listdir', return_value=['pre_gen_project.py']), patch('valid_hook', return_value=True):
        result = find_hook('pre_gen_project')
    assert result is not None


# LLM-generated content at query #13
#--------------------------

def test_find_hook_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_empty_hooks_dir():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_no_matching_script():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        open(os.path.join(hooks_dir, 'other_hook.py'), 'w').close()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        pass
    assert not os.path.exists(project_dir)


# LLM-generated content at query #15
#--------------------------

def test_find_hook_with_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_with_empty_hooks_dir(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_valid_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project.py')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file)]

def test_find_hook_with_backup_file(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project.py~')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_unsupported_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('unsupported_hook.py')
    hook_file.write('')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

def test_find_hook_with_mismatched_name(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('post_gen_project.py')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_with_multiple_valid_hooks(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file1 = hooks_dir.join('pre_gen_project.py')
    hook_file1.write('')
    hook_file2 = hooks_dir.join('post_gen_project.py')
    hook_file2.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file1)]

def test_find_hook_without_extension(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None


# LLM-generated content at query #16
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    result = find_hook('pre_gen_project', 'hooks')
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0


# LLM-generated content at query #17
#--------------------------

def test_find_hook_returns_list_when_valid_hook_exists():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'pre_gen_project.py'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
                assert isinstance(result, list)
                assert len(result) == 1
                assert os.path.basename(result[0]) == hook_file

def test_find_hook_returns_none_when_hooks_dir_not_exist():
    import os
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'non_existent_hooks')
        assert result is None

def test_find_hook_returns_none_when_no_valid_hooks():
    import os
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['invalid_hook.txt']):
            with patch('valid_hook', return_value=False):
                result = find_hook('pre_gen_project', 'hooks')
                assert result is None

def test_find_hook_returns_absolute_paths():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = 'post_gen_project.py'
        with open(os.path.join(hooks_dir, hook_file), 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file]):
            with patch('valid_hook', return_value=True):
                result = find_hook('post_gen_project', hooks_dir)
                assert os.path.isabs(result[0])

def test_find_hook_filters_by_valid_hook():
    import os
    from unittest.mock import patch
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=['pre_gen_project.py', 'invalid.txt', 'post_gen_project.sh']):
            with patch('valid_hook') as mock_valid:
                mock_valid.side_effect = lambda f, h: f.endswith('.py')
                result = find_hook('pre_gen_project', 'hooks')
                assert len(result) == 1
                assert result[0].endswith('pre_gen_project.py')


# LLM-generated content at query #18
#--------------------------

def test_no_hook_found_when_scripts_is_empty():
    scripts = []
    result = not scripts
    assert result is True


# LLM-generated content at query #19
#--------------------------

def test_find_hook_no_scripts_found():
    result = find_hook('post_gen_project', 'non_existent_hooks_dir')
    assert result is None


# LLM-generated content at query #20
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda x: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script, cwd, ctx: None
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.work_in = lambda dirname: contextlib.nullcontext()
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_no_hook():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda x: None
    hooks.find_hook = mock_find_hook
    utils.work_in = lambda dirname: contextlib.nullcontext()
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_hook_fails_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda x: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script, cwd, ctx: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.work_in = lambda dirname: contextlib.nullcontext()
    utils.rmtree = lambda path: None
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except hooks.FailedHookException:
        pass

def test_run_hook_from_repo_dir_hook_fails_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    mock_find_hook = lambda x: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script, cwd, ctx: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.work_in = lambda dirname: contextlib.nullcontext()
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except hooks.FailedHookException:
        pass

def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda x: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script, cwd, ctx: (_ for _ in ()).throw(jinja2.UndefinedError('Undefined variable'))
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.work_in = lambda dirname: contextlib.nullcontext()
    utils.rmtree = lambda path: None
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except jinja2.UndefinedError:
        pass


# LLM-generated content at query #21
#--------------------------

def test_run_script_successful_python_script():
    import tempfile
    import os
    from pathlib import Path
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_successful_non_python_script():
    import tempfile
    import os
    from pathlib import Path
    script_content = "#!/bin/sh\nexit 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_failed_exit_status():
    import tempfile
    import os
    from pathlib import Path
    script_content = "import sys; sys.exit(1)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_enoexec_error():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("")
        script_path = f.name
    os.chmod(script_path, 0o444)
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error():
    import tempfile
    import os
    from pathlib import Path
    script_path = "/non/existent/path/script.py"
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)

def test_run_script_with_cwd():
    import tempfile
    import os
    from pathlib import Path
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script_content)
        script_path = f.name
    cwd = Path('/tmp')
    try:
        run_script(script_path, cwd=cwd)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = "/fake/script.py"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = "rendered content"
    mock_env.from_string.return_value = mock_template
    mock_temp_file = Mock()
    mock_temp_file.name = "/fake/temp.py"
    with patch("cookiecutter.hooks.create_env_with_context", return_value=mock_env) as mock_create_env, patch("tempfile.NamedTemporaryFile", return_value=mock_temp_file) as mock_temp, patch("cookiecutter.hooks.Path") as mock_path, patch("cookiecutter.hooks.run_script") as mock_run_script:
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "original content"
        cookiecutter.hooks.run_script_with_context(script_path, cwd, context)
        mock_create_env.assert_called_once_with(context)
        mock_path.assert_called_once_with(script_path)
        mock_path_instance.read_text.assert_called_once_with(encoding="utf-8")
        mock_env.from_string.assert_called_once_with("original content")
        mock_template.render.assert_called_once_with(**context)
        mock_temp.assert_called_once_with(delete=False, mode="wb", suffix=".py")
        mock_temp_file.write.assert_called_once_with(b"rendered content")
        mock_run_script.assert_called_once_with("/fake/temp.py", cwd)


# LLM-generated content at query #23
#--------------------------

def test_find_hook_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_empty_hooks_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_no_matching_scripts():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'other_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None


# LLM-generated content at query #24
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook', side_effect=cookiecutter.hooks.FailedHookException('Hook failed')):
        with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
            with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                try:
                    cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                except cookiecutter.hooks.FailedHookException:
                    pass
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_failed_hook_exception_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = False
    with unittest.mock.patch('cookiecutter.hooks.run_hook', side_effect=cookiecutter.hooks.FailedHookException('Hook failed')):
        with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
            with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                try:
                    cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                except cookiecutter.hooks.FailedHookException:
                    pass
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook', side_effect=cookiecutter.exceptions.UndefinedError('Undefined variable')):
        with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
            with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                try:
                    cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                except cookiecutter.exceptions.UndefinedError:
                    pass
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_work_in_context():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
        with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #25
#--------------------------

def test_run_hook_no_scripts_found():
    from cookiecutter.hooks import run_hook
    from pathlib import Path
    import logging
    logger = logging.getLogger('cookiecutter')
    scripts = []
    context = {}
    project_dir = Path('.')
    run_hook('pre_gen_project', project_dir, context)
    assert logger.debug.called
    assert 'No pre_gen_project hook found' in logger.debug.call_args[0]


# LLM-generated content at query #26
#--------------------------

def test_find_hook_no_scripts_found():
    result = find_hook('post_gen_project', 'empty_hooks_dir')
    assert result is None


# LLM-generated content at query #27
#--------------------------

def test_oserror_errno_not_enoexec():
    err = OSError()
    err.errno = 0
    result = err.errno == errno.ENOEXEC
    assert result is False


# LLM-generated content at query #28
#--------------------------

def test_run_script_successful_python_script():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    finally:
        os.unlink(script_path)

def test_run_script_successful_non_python_script():
    import tempfile
    import os
    import sys
    import stat
    from pathlib import Path
    script_content = "#!/bin/sh\nexit 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    finally:
        os.unlink(script_path)

def test_run_script_fails_with_non_zero_exit():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_content = "import sys; sys.exit(1)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_fails_with_enoexec():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_content = ""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_fails_with_oserror():
    import tempfile
    import os
    import sys
    from pathlib import Path
    non_existent_cwd = "/non/existent/path"
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path, cwd=non_existent_cwd)
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #29
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_no_hooks_directory():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py~')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_unsupported_hook_name():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'unsupported_hook.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('unsupported_hook', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result is None

def test_find_hook_with_matching_hook_but_different_extension():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.sh')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_multiple_valid_hooks():
    import tempfile, os, shutil
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path1 = os.path.join(hooks_dir, 'pre_gen_project.py')
    hook_file_path2 = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_file_path1, 'w') as f:
        f.write('')
    with open(hook_file_path2, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(hook_file_path1)]


# LLM-generated content at query #30
#--------------------------

```python
def test_run_script_with_context_creates_non_deletable_temp_file():
    script_path = "/fake/path/script.py"
    cwd = "/fake/cwd"
    context = {"cookiecutter": {"key": "value"}}
    temp_file_name = None
    try:
        run_script_with_context(script_path, cwd, context)
        temp_file_name = glob.glob("/tmp/*")[0]
        os.remove(temp_file_name)
        assert False, "File should not be deletable immediately"
    except PermissionError:
        assert True
    except FileNotFoundError:
        assert False, "File should exist after creation"
    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


