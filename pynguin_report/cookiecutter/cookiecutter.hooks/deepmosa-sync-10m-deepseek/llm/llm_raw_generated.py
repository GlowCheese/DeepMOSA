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

def test_valid_hook_nonmatching_supported_not_backup():
    result = valid_hook("/some/path/real_hook.py", "different_hook")
    assert result == False

def test_valid_hook_nonmatching_supported_backup():
    result = valid_hook("/some/path/real_hook.py~", "different_hook")
    assert result == False

def test_valid_hook_nonmatching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown.py", "hook_name")
    assert result == False

def test_valid_hook_nonmatching_unsupported_backup():
    result = valid_hook("/some/path/unknown.py~", "hook_name")
    assert result == False

def test_valid_hook_matching_unsupported_backup():
    result = valid_hook("/some/path/unknown_hook.py~", "unknown_hook")
    assert result == False


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

def test_run_hook_no_scripts_found():
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

def test_run_hook_with_jinja_script():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_gen_project.py'
        script_path.write_text('print("{{ cookiecutter.project_name }}")')
        project_dir = Path(tmpdir)
        context = {'cookiecutter': {'project_name': 'TestProject'}}
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

def test_run_hook_with_multiple_scripts():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_gen_project.py'
        script1.write_text('print("First")')
        script2 = hooks_dir / 'pre_gen_project.sh'
        script2.write_text('echo "Second"')
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)

def test_run_hook_with_non_py_script():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_gen_project.sh'
        script_path.write_text('#!/bin/bash\necho "Hello"')
        os.chmod(script_path, 0o755)
        project_dir = Path(tmpdir)
        context = {}
        run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #4
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/some/path/hook_name.py", "hook_name")
    assert result == True


# LLM-generated content at query #5
#--------------------------

def test_run_pre_prompt_hook_with_no_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'pre_prompt.py'
        hook_script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert isinstance(result, Path)

def test_run_pre_prompt_hook_with_failing_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'pre_prompt.py'
        hook_script.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException as e:
            assert 'Pre-Prompt Hook script failed' in str(e)

def test_run_pre_prompt_hook_with_empty_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'pre_prompt.py'
        hook_script.write_text('')
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException as e:
            assert 'Hook script failed' in str(e)

def test_run_pre_prompt_hook_with_multiple_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script1 = hooks_dir / 'pre_prompt.py'
        hook_script1.write_text('import sys\nsys.exit(0)')
        hook_script2 = hooks_dir / 'pre_prompt.sh'
        hook_script2.write_text('#!/bin/bash\nexit 0')
        hook_script2.chmod(0o755)
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert isinstance(result, Path)

def test_run_pre_prompt_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'pre_prompt.py~'
        hook_script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_unsupported_hook_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'unsupported_hook.py'
        hook_script.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_without_hooks_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = "/some/path"
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #7
#--------------------------

def test_no_hook_found():
    scripts = []
    result = not scripts
    assert result == True


# LLM-generated content at query #8
#--------------------------

def test_hooks_dir_is_not_a_directory():
    os.path.isdir = lambda x: False


# LLM-generated content at query #9
#--------------------------

def test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir():
    repo_dir = Path('/some/repo')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #10
#--------------------------

def test_run_script_successful_python_script():
    import tempfile
    import os
    import sys
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_successful_shell_script():
    import tempfile
    import os
    import sys
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
    import sys
    script_content = "import sys; sys.exit(1)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_enoexec_error():
    import tempfile
    import os
    import sys
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("")
        script_path = f.name
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error():
    import os
    import sys
    script_path = "/non/existent/path/script.py"
    try:
        run_script(script_path)
        assert False, "Expected FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)

def test_run_script_with_cwd():
    import tempfile
    import os
    import sys
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    cwd = os.path.dirname(script_path)
    try:
        run_script(script_path, cwd=cwd)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #11
#--------------------------

def test_run_pre_prompt_hook_with_no_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()

def test_run_pre_prompt_hook_with_failing_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException as e:
            assert 'Pre-Prompt Hook script failed' in str(e)

def test_run_pre_prompt_hook_with_invalid_hook_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.sh'
        hook_file.write_text('#!/bin/bash\nexit 0')
        hook_file.chmod(0o755)
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()

def test_run_pre_prompt_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        valid_hook = hooks_dir / 'pre_prompt.py'
        valid_hook.write_text('import sys\nsys.exit(0)')
        backup_hook = hooks_dir / 'pre_prompt.py~'
        backup_hook.write_text('import sys\nsys.exit(1)')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()

def test_run_pre_prompt_hook_with_wrong_hook_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_empty_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('')
        try:
            run_pre_prompt_hook(repo_dir)
            assert False
        except FailedHookException as e:
            assert 'Hook script failed' in str(e)

def test_run_pre_prompt_hook_with_multiple_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook1 = hooks_dir / 'pre_prompt.py'
        hook1.write_text('import sys\nsys.exit(0)')
        hook2 = hooks_dir / 'pre_prompt.sh'
        hook2.write_text('#!/bin/bash\nexit 0')
        hook2.chmod(0o755)
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()

def test_run_pre_prompt_hook_with_no_hooks_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #12
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_hook = lambda hook_name, project_dir, context: None
    cookiecutter.hooks.find_hook = mock_find_hook
    cookiecutter.hooks.run_hook = mock_run_hook
    cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_no_hook_found():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: None
    cookiecutter.hooks.find_hook = mock_find_hook
    cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_hook_fails_with_failedhookexception():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_hook = lambda hook_name, project_dir, context: exec('raise FailedHookException("Hook failed")')
    cookiecutter.hooks.find_hook = mock_find_hook
    cookiecutter.hooks.run_hook = mock_run_hook
    mock_rmtree = lambda path: None
    cookiecutter.utils.rmtree = mock_rmtree
    try:
        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass

def test_run_hook_from_repo_dir_hook_fails_with_undefinederror():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_hook = lambda hook_name, project_dir, context: exec('raise UndefinedError("Undefined variable")')
    cookiecutter.hooks.find_hook = mock_find_hook
    cookiecutter.hooks.run_hook = mock_run_hook
    mock_rmtree = lambda path: None
    cookiecutter.utils.rmtree = mock_rmtree
    try:
        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except UndefinedError:
        pass

def test_run_hook_from_repo_dir_hook_fails_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    mock_find_hook = lambda hook_name: ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_hook = lambda hook_name, project_dir, context: exec('raise FailedHookException("Hook failed")')
    cookiecutter.hooks.find_hook = mock_find_hook
    cookiecutter.hooks.run_hook = mock_run_hook
    mock_rmtree = lambda path: None
    cookiecutter.utils.rmtree = mock_rmtree
    try:
        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass


# LLM-generated content at query #13
#--------------------------

def test_hooks_dir_is_not_a_directory():
    hooks_dir = 'non_existent_dir'
    result = find_hook('some_hook', hooks_dir)
    assert result is None


# LLM-generated content at query #14
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/some/path/hook_name.py", "hook_name")
    assert result == True


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
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                        with patch('valid_hook', return_value=True):
                            result = find_hook('pre_gen_project', hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, hook_file))]

def test_find_hook_with_no_hooks_directory():
    with patch('os.path.isdir', return_value=False):
        with patch('logger.debug') as mock_debug:
            result = find_hook('pre_gen_project', 'hooks')
    assert result is None
    mock_debug.assert_called_with('No hooks/dir in template_dir')

def test_find_hook_with_empty_scripts_list():
    with patch('os.path.isdir', return_value=True):
        with patch('os.listdir', return_value=[]):
            with patch('valid_hook', return_value=False):
                result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_multiple_valid_scripts():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_files = ['pre_gen_project.py', 'post_gen_project.py']
        for hf in hook_files:
            with open(os.path.join(hooks_dir, hf), 'w') as f:
                f.write('')
        with patch('os.listdir', return_value=hook_files):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', side_effect=lambda x: x):
                    with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                        with patch('valid_hook', side_effect=lambda f, n: f.startswith('pre')):
                            result = find_hook('pre_gen_project', hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py'))]


# LLM-generated content at query #17
#--------------------------

def test_run_hook_no_hooks_dir():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch
    temp_dir = tempfile.mkdtemp()
    context = {'cookiecutter': {}}
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = None
        run_hook('pre_gen_project', temp_dir, context)
        mock_find.assert_called_once_with('pre_gen_project')
    os.rmdir(temp_dir)

def test_run_hook_with_scripts():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    temp_dir = tempfile.mkdtemp()
    context = {'cookiecutter': {}}
    mock_script = os.path.join(temp_dir, 'pre_gen_project.py')
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            mock_find.return_value = [mock_script]
            run_hook('pre_gen_project', temp_dir, context)
            mock_find.assert_called_once_with('pre_gen_project')
            mock_run.assert_called_once_with(mock_script, temp_dir, context)
    os.rmdir(temp_dir)

def test_run_hook_multiple_scripts():
    import tempfile
    import os
    from cookiecutter.hooks import run_hook
    from unittest.mock import patch, MagicMock
    temp_dir = tempfile.mkdtemp()
    context = {'cookiecutter': {}}
    mock_script1 = os.path.join(temp_dir, 'pre_gen_project.py')
    mock_script2 = os.path.join(temp_dir, 'pre_gen_project.sh')
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            mock_find.return_value = [mock_script1, mock_script2]
            run_hook('pre_gen_project', temp_dir, context)
            mock_find.assert_called_once_with('pre_gen_project')
            assert mock_run.call_count == 2
            mock_run.assert_any_call(mock_script1, temp_dir, context)
            mock_run.assert_any_call(mock_script2, temp_dir, context)
    os.rmdir(temp_dir)


# LLM-generated content at query #18
#--------------------------

def test_no_hook_found_when_scripts_is_empty():
    from cookiecutter.hooks import run_hook
    from pathlib import Path
    import logging
    logger = logging.getLogger('cookiecutter')
    scripts = []
    context = {'cookiecutter': {'project_name': 'Test'}}
    project_dir = Path('/tmp/test')
    run_hook('pre_gen_project', project_dir, context)
    assert logger.debug.called
    assert 'No pre_gen_project hook found' in logger.debug.call_args[0]


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_with_delete_flag():
    mock_repo_dir = "/fake/repo"
    mock_project_dir = "/fake/project"
    mock_context = {"cookiecutter": {"name": "test"}}
    
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
        with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                    mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
                    
                    try:
                        cookiecutter.hooks.run_hook_from_repo_dir(
                            repo_dir=mock_repo_dir,
                            hook_name="pre_gen_project",
                            project_dir=mock_project_dir,
                            context=mock_context,
                            delete_project_on_failure=True
                        )
                    except cookiecutter.hooks.FailedHookException:
                        pass
                    
                    mock_rmtree.assert_called_once_with(mock_project_dir)


# LLM-generated content at query #20
#--------------------------

def test_find_hook_with_valid_hook_in_hooks_dir():
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
            with patch('valid_hook', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_no_hooks_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
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

def test_find_hook_with_no_matching_hook():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=['some_file.txt']):
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
        hook_file1 = 'pre_gen_project.py'
        hook_file2 = 'pre_gen_project.sh'
        hook_file_path1 = os.path.join(hooks_dir, hook_file1)
        hook_file_path2 = os.path.join(hooks_dir, hook_file2)
        with open(hook_file_path1, 'w') as f:
            f.write('')
        with open(hook_file_path2, 'w') as f:
            f.write('')
        with patch('os.listdir', return_value=[hook_file1, hook_file2]):
            with patch('valid_hook', side_effect=lambda f, n: f in [hook_file1, hook_file2]):
                result = find_hook('pre_gen_project', hooks_dir)
    expected = sorted([os.path.abspath(hook_file_path1), os.path.abspath(hook_file_path2)])
    assert sorted(result) == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = "/fake/script.py.j2"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_rendered_content = "print('Hello TestProject')"
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "print('Hello {{ cookiecutter.project_name }}')"
        
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_template = Mock()
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = mock_rendered_content
        
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_path_class.assert_called_once_with(mock_script_path)
        mock_path_instance.read_text.assert_called_once_with(encoding='utf-8')
        mock_create_env.assert_called_once_with(mock_context)
        mock_env.from_string.assert_called_once_with("print('Hello {{ cookiecutter.project_name }}')")
        mock_template.render.assert_called_once_with(**mock_context)
        mock_temp.write.assert_called_once_with(mock_rendered_content.encode('utf-8'))
        mock_run_script.assert_called_once_with("/fake/temp.py", mock_cwd)


def test_run_script_with_context_preserves_file_extension():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = "/fake/script.sh.j2"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "echo 'Hello {{ cookiecutter.project_name }}'"
        
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_template = Mock()
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "echo 'Hello TestProject'"
        
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.sh"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.sh')


def test_run_script_with_context_handles_empty_extension():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = "/fake/script.j2"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "content without extension"
        
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_template = Mock()
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "rendered content"
        
        mock_temp = Mock()
        mock_temp.name = "/fake/temp"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='')


def test_run_script_with_context_passes_context_to_template_render():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = "/fake/script.py.j2"
    mock_cwd = "/fake/cwd"
    mock_context = {
        "cookiecutter": {
            "project_name": "TestProject",
            "version": "1.0.0"
        },
        "extra_var": "value"
    }
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "template content"
        
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_template = Mock()
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "rendered content"
        
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_template.render.assert_called_once_with(**mock_context)


def test_run_script_with_context_encodes_output_as_utf8():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = "/fake/script.py.j2"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_rendered_content = "print('Hello TestProject')"
    
    with patch('cookiecutter.hooks.Path') as mock_path_class, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.read_text.return_value = "template content"
        
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_template = Mock()
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = mock_rendered_content
        
        mock_temp = Mock()
        mock_temp.name = "/fake/temp.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        expected_bytes = mock_rendered_content.encode('utf-8')
        mock_temp.write.assert_called_once_with(expected_bytes)


# LLM-generated content at query #22
#--------------------------

def test_find_hook_with_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_with_empty_hooks_dir(mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=[])
    result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_invalid_hook_files(mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['invalid.txt', 'backup~'])
    mocker.patch('os.path.abspath', side_effect=lambda x: f'/abs/{x}')
    result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_valid_hook_file(mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['pre_gen_project.py'])
    mocker.patch('os.path.abspath', side_effect=lambda x: f'/abs/{x}')
    mocker.patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    result = find_hook('pre_gen_project', 'hooks')
    assert result == ['/abs/hooks/pre_gen_project.py']

def test_find_hook_with_multiple_valid_hook_files(mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['pre_gen_project.py', 'post_gen_project.py'])
    mocker.patch('os.path.abspath', side_effect=lambda x: f'/abs/{x}')
    mocker.patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    result = find_hook('pre_gen_project', 'hooks')
    assert result == ['/abs/hooks/pre_gen_project.py']


# LLM-generated content at query #23
#--------------------------

def test_find_hook_with_valid_hook():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result == [hook_file_path]

def test_find_hook_with_no_hooks_directory():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(hook_file_path, 'w') as f:
            f.write('')
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('unsupported_hook', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file_path1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file_path2 = os.path.join(hooks_dir, 'post_gen_project.py')
        with open(hook_file_path1, 'w') as f:
            f.write('')
        with open(hook_file_path2, 'w') as f:
            f.write('')
        with patch('os.path.abspath', return_value=tmpdir):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result == [hook_file_path1]


# LLM-generated content at query #24
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
    has_py_suffix = any(f.suffix == ".py" for f in temp_files)
    
    assert has_py_suffix


# LLM-generated content at query #25
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test"}}
    delete_project_on_failure = False
    with work_in(repo_dir):
        try:
            run_hook(hook_name, project_dir, context)
        except (FailedHookException, UndefinedError):
            if delete_project_on_failure:
                rmtree(project_dir)
            logger.exception("Stopping generation because %s hook script didn't exit successfully", hook_name)
            raise


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook_without_scripts_returns_original_repo_dir():
    repo_dir = Path('/some/test/dir')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #27
#--------------------------

def test_find_hook_with_nonexistent_hooks_dir():
    result = find_hook('pre_gen_project', 'nonexistent_dir')
    assert result is None

def test_find_hook_with_empty_hooks_dir():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None

def test_find_hook_with_no_matching_scripts():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'post_gen_project.py'), 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_run_script_with_context_creates_temporary_file_with_correct_extension():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import create_env_with_context
    from cookiecutter.hooks import run_script_with_context
    
    test_script_path = "test_script.py"
    test_cwd = "/tmp"
    test_context = {"cookiecutter": {"project_name": "test"}}
    
    original_read_text = Path.read_text
    mock_content = "print('{{ cookiecutter.project_name }}')"
    
    def mock_read_text(self, encoding=None):
        return mock_content
    
    Path.read_text = mock_read_text
    
    try:
        run_script_with_context(test_script_path, test_cwd, test_context)
        
        temp_files = [f for f in os.listdir(tempfile.gettempdir()) 
                     if f.startswith('tmp') and f.endswith('.py')]
        
        assert len(temp_files) > 0
        
        for temp_file in temp_files:
            temp_path = os.path.join(tempfile.gettempdir(), temp_file)
            if os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    content = f.read()
                    assert b"print('test')" in content
                os.unlink(temp_path)
                break
    finally:
        Path.read_text = original_read_text


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = '/test/script.py.j2'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered content'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text', return_value='template content'), \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp123.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_create_env.assert_called_once_with(context)
        mock_env.from_string.assert_called_once_with('template content')
        mock_template.render.assert_called_once_with(**context)
        mock_temp.write.assert_called_once_with(b'rendered content')
        mock_run_script.assert_called_once_with('/tmp/temp123.py', cwd)

def test_run_script_with_context_preserves_file_extension():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = '/test/script.sh.j2'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered bash script'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text', return_value='bash template'), \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp456.sh'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
        mock_temp.write.assert_called_once_with(b'rendered bash script')
        mock_run_script.assert_called_once_with('/tmp/temp456.sh', cwd)

def test_run_script_with_context_handles_empty_extension():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = '/test/script'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered content'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text', return_value='template'), \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp789'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='')
        mock_temp.write.assert_called_once_with(b'rendered content')
        mock_run_script.assert_called_once_with('/tmp/temp789', cwd)

def test_run_script_with_context_uses_utf8_encoding():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = '/test/script.py.j2'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered content with unicode: éàü'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text', return_value='template'), \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp999.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_temp.write.assert_called_once_with(b'rendered content with unicode: \xc3\xa9\xc3\xa0\xc3\xbc')

def test_run_script_with_context_passes_context_to_template_render():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    script_path = '/test/script.py.j2'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'project_name': 'TestProject', 'version': '1.0'}, 'extra': 'value'}
    mock_env = Mock()
    mock_template = Mock()
    mock_template.render.return_value = 'rendered'
    mock_env.from_string.return_value = mock_template
    with patch('cookiecutter.hooks.create_env_with_context', return_value=mock_env) as mock_create_env, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('pathlib.Path.read_text', return_value='template'), \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp = Mock()
        mock_temp.name = '/tmp/temp111.py'
        mock_temp.__enter__.return_value = mock_temp
        mock_temp_file.return_value = mock_temp
        from cookiecutter.hooks import run_script_with_context
        run_script_with_context(script_path, cwd, context)
        mock_template.render.assert_called_once_with(**context)


# LLM-generated content at query #30
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook('/path/to/hook.py', 'hook')
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook('/path/to/hook.py~', 'hook')
    assert result == False

def test_valid_hook_matching_unsupported_not_backup():
    result = valid_hook('/path/to/unknown.py', 'unknown')
    assert result == False

def test_valid_hook_nonmatching_supported_not_backup():
    result = valid_hook('/path/to/other_hook.py', 'hook')
    assert result == False

def test_valid_hook_nonmatching_unsupported_not_backup():
    result = valid_hook('/path/to/unknown.py', 'hook')
    assert result == False

def test_valid_hook_nonmatching_supported_backup():
    result = valid_hook('/path/to/other_hook.py~', 'hook')
    assert result == False

def test_valid_hook_matching_unsupported_backup():
    result = valid_hook('/path/to/unknown.py~', 'unknown')
    assert result == False

def test_valid_hook_nonmatching_unsupported_backup():
    result = valid_hook('/path/to/unknown.py~', 'hook')
    assert result == False


# LLM-generated content at query #31
#--------------------------

def test_predicate_at_line_21_evaluates_to_false():
    err = OSError()
    err.errno = 0
    predicate_result = err.errno == errno.ENOEXEC
    assert predicate_result is False


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_when_flag_true():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test Project"}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in, \
         unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
        mock_work_in.return_value.__enter__.return_value = None
        mock_work_in.return_value.__exit__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #33
#--------------------------

def test_predicate_at_line_21_evaluates_to_false():
    import subprocess
    import sys
    import errno
    from pathlib import Path
    from unittest.mock import Mock, patch, call
    from pre_commit.error import FailedHookException
    from pre_commit.util import make_executable
    from pre_commit.runner import run_script
    EXIT_SUCCESS = 0
    script_path = '/tmp/test_script.py'
    cwd = Path('.')
    run_thru_shell = sys.platform.startswith('win')
    script_command = [sys.executable, script_path]
    mock_proc = Mock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, patch('pre_commit.util.make_executable') as mock_make_executable:
        run_script(script_path, cwd)
        mock_make_executable.assert_called_once_with(script_path)
        mock_popen.assert_called_once_with(script_command, shell=run_thru_shell, cwd=cwd)
        mock_proc.wait.assert_called_once()
    mock_proc = Mock()
    mock_proc.wait.return_value = 1
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, patch('pre_commit.util.make_executable') as mock_make_executable:
        try:
            run_script(script_path, cwd)
        except FailedHookException as e:
            assert str(e) == 'Hook script failed (exit status: 1)'
    mock_proc = Mock()
    mock_proc.wait.side_effect = OSError(errno.ENOENT, 'No such file or directory')
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, patch('pre_commit.util.make_executable') as mock_make_executable:
        try:
            run_script(script_path, cwd)
        except FailedHookException as e:
            assert 'Hook script failed (error:' in str(e)
    mock_proc = Mock()
    mock_proc.wait.side_effect = OSError(errno.ENOEXEC, 'Exec format error')
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen, patch('pre_commit.util.make_executable') as mock_make_executable:
        try:
            run_script(script_path, cwd)
        except FailedHookException as e:
            assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'
    assert True


# LLM-generated content at query #34
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
        run_script(script_path, cwd=os.path.dirname(script_path))
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
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
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
        run_script(script_path, cwd=os.path.dirname(script_path))
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error_enoexec():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("")
        script_path = f.name
    os.chmod(script_path, 0o644)
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error_generic():
    import tempfile
    import os
    from pathlib import Path
    script_path = "/non/existent/path/script.py"
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #35
#--------------------------

def test_find_hook_no_hooks_dir():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_no_matching_scripts():
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, 'unrelated.txt'), 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', temp_dir)
    assert result is None

def test_find_hook_empty_hooks_dir():
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    result = find_hook('pre_gen_project', temp_dir)
    assert result is None


# LLM-generated content at query #36
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_when_configured():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test"}}
    delete_project_on_failure = True
    mock_work_in = unittest.mock.MagicMock()
    mock_run_hook = unittest.mock.MagicMock(side_effect=FailedHookException("Hook failed"))
    mock_rmtree = unittest.mock.MagicMock()
    with unittest.mock.patch('cookiecutter.hooks.work_in', mock_work_in):
        with unittest.mock.patch('cookiecutter.hooks.run_hook', mock_run_hook):
            with unittest.mock.patch('cookiecutter.hooks.rmtree', mock_rmtree):
                try:
                    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                except FailedHookException:
                    pass
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #37
#--------------------------

def test_find_hook_with_valid_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    with patch('os.listdir', return_value=['pre_gen_project.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    expected = [os.path.abspath(hook_file_path)]
    assert result == expected

def test_find_hook_with_no_hooks_dir():
    import os
    from unittest.mock import patch
    with patch('os.path.isdir', return_value=False):
        result = find_hook('pre_gen_project', 'hooks')
    assert result is None

def test_find_hook_with_empty_hooks_dir():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=[]):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_invalid_hook_file():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['invalid_hook.txt']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['pre_gen_project.py~']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['unsupported_hook.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('unsupported_hook', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_files():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file1_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    hook_file2_path = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_file1_path, 'w') as f:
        f.write('')
    with open(hook_file2_path, 'w') as f:
        f.write('')
    with patch('os.listdir', return_value=['pre_gen_project.py', 'post_gen_project.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    expected = [os.path.abspath(hook_file1_path)]
    assert result == expected


# LLM-generated content at query #38
#--------------------------

def test_run_script_successful_python():
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

def test_run_script_successful_non_python():
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
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error():
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    cwd = Path.home()
    try:
        run_script(script_path, cwd)
    finally:
        os.unlink(script_path)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_script_with_context_creates_temporary_file_with_correct_suffix():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.utils import create_env_with_context
    from cookiecutter.hooks import run_script_with_context
    
    test_script_path = "/tmp/test_script.py"
    test_cwd = "/tmp"
    test_context = {"cookiecutter": {"project_name": "Test Project"}}
    
    Path(test_script_path).write_text("print('{{ cookiecutter.project_name }}')", encoding='utf-8')
    
    run_script_with_context(test_script_path, test_cwd, test_context)
    
    temp_files = list(Path(tempfile.gettempdir()).glob("*.py"))
    assert any(temp_file.suffix == ".py" for temp_file in temp_files)
    
    os.remove(test_script_path)
    for temp_file in temp_files:
        if temp_file.name.startswith("tmp"):
            os.remove(temp_file)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_on_failure_when_flag_false():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False
    import sys
    import types
    from unittest.mock import Mock, patch
    mock_work_in = Mock()
    mock_work_in.__enter__ = Mock()
    mock_work_in.__exit__ = Mock()
    mock_rmtree = Mock()
    mock_logger = Mock()
    mock_run_hook = Mock(side_effect=Exception("hook failed"))
    sys.modules['cookiecutter.utils'] = types.ModuleType('utils')
    sys.modules['cookiecutter.utils'].work_in = Mock(return_value=mock_work_in)
    sys.modules['cookiecutter.utils'].rmtree = mock_rmtree
    sys.modules['cookiecutter.hooks'] = types.ModuleType('hooks')
    sys.modules['cookiecutter.hooks'].run_hook = mock_run_hook
    sys.modules['cookiecutter.hooks'].FailedHookException = Exception
    sys.modules['cookiecutter.hooks'].UndefinedError = Exception
    sys.modules['cookiecutter.hooks'].logger = mock_logger
    from cookiecutter.hooks import run_hook_from_repo_dir
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except Exception:
        pass
    mock_rmtree.assert_not_called()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_run_script_success_python():
    import tempfile
    import os
    from pathlib import Path
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path, cwd=Path('.'))
    finally:
        os.unlink(script_path)

def test_run_script_success_non_python():
    import tempfile
    import os
    from pathlib import Path
    script_content = "#!/bin/sh\nexit 0"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path, cwd=Path('.'))
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
        run_script(script_path, cwd=Path('.'))
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error_enoexec():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("")
        script_path = f.name
    os.chmod(script_path, 0o644)
    try:
        run_script(script_path, cwd=Path('.'))
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)

def test_run_script_os_error_generic():
    import tempfile
    import os
    from pathlib import Path
    script_path = "/non/existent/path/script.sh"
    try:
        run_script(script_path, cwd=Path('.'))
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)

def test_run_script_cwd():
    import tempfile
    import os
    from pathlib import Path
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


# LLM-generated content at query #2
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/path/to/hook.py", "hook")
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook("/path/to/hook.py~", "hook")
    assert result == False

def test_valid_hook_matching_unsupported():
    result = valid_hook("/path/to/unknown.py", "unknown")
    assert result == False

def test_valid_hook_not_matching_supported():
    result = valid_hook("/path/to/other.py", "hook")
    assert result == False

def test_valid_hook_not_matching_unsupported():
    result = valid_hook("/path/to/unknown.py", "hook")
    assert result == False

def test_valid_hook_empty_hook_name():
    result = valid_hook("/path/to/.py", "")
    assert result == False

def test_valid_hook_file_without_extension():
    result = valid_hook("/path/to/hook", "hook")
    assert result == True

def test_valid_hook_file_with_multiple_dots():
    result = valid_hook("/path/to/hook.test.py", "hook.test")
    assert result == False


# LLM-generated content at query #3
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/some/path/hook_name.py", "hook_name")
    assert result == True


# LLM-generated content at query #4
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
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_no_matching_hook():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        other_hook = hooks_dir / 'other_hook.py'
        other_hook.write_text('')
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_backup_file():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        backup_hook = hooks_dir / 'pre_gen_project.py~'
        backup_hook.write_text('')
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_unsupported_hook_name():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        unsupported_hook = hooks_dir / 'unsupported_hook.py'
        unsupported_hook.write_text('')
        context = {}
        run_hook('unsupported_hook', project_dir, context)


def test_run_hook_with_valid_hook():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        valid_hook = hooks_dir / 'pre_gen_project.py'
        valid_hook.write_text('print("Hello from hook")')
        context = {}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_jinja_template():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        template_hook = hooks_dir / 'pre_gen_project.py'
        template_hook.write_text('print("{{ cookiecutter.project_name }}")')
        context = {'cookiecutter': {'project_name': 'TestProject'}}
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_multiple_valid_hooks():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        hooks_dir = project_dir / 'hooks'
        hooks_dir.mkdir()
        hook1 = hooks_dir / 'pre_gen_project.py'
        hook1.write_text('print("Hook 1")')
        hook2 = hooks_dir / 'pre_gen_project.sh'
        hook2.write_text('echo "Hook 2"')
        context = {}
        run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #5
#--------------------------

def test_find_hook_with_valid_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    with patch('os.listdir', return_value=['pre_gen_project.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    expected = [os.path.abspath(hook_file_path)]
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
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=[]):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['pre_gen_project.py~']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['unsupported_hook.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('unsupported_hook', hooks_dir)
    assert result is None

def test_find_hook_with_mismatched_hook_name():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    with patch('os.listdir', return_value=['post_gen_project.py']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    test_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(test_dir, 'hooks')
    os.makedirs(hooks_dir)
    hook_file1_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    hook_file2_path = os.path.join(hooks_dir, 'pre_gen_project.sh')
    with open(hook_file1_path, 'w') as f:
        f.write('')
    with open(hook_file2_path, 'w') as f:
        f.write('')
    with patch('os.listdir', return_value=['pre_gen_project.py', 'pre_gen_project.sh']):
        with patch('os.path.isdir', return_value=True):
            result = find_hook('pre_gen_project', hooks_dir)
    expected = [os.path.abspath(hook_file1_path), os.path.abspath(hook_file2_path)]
    assert sorted(result) == sorted(expected)


# LLM-generated content at query #6
#--------------------------

def test_no_hook_found_when_scripts_is_empty():
    scripts = []
    result = not scripts
    assert result is True


# LLM-generated content at query #7
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__.return_value = None
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
            mock_work_in.assert_called_once_with(repo_dir)
            mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_hook_failure_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException('Hook failed')
        with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__.return_value = None
            with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                    try:
                        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                    except cookiecutter.hooks.FailedHookException:
                        pass
                    mock_work_in.assert_called_once_with(repo_dir)
                    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
                    mock_rmtree.assert_called_once_with(project_dir)
                    mock_logger.exception.assert_called_once()

def test_run_hook_from_repo_dir_hook_failure_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException('Hook failed')
        with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__.return_value = None
            with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                    try:
                        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                    except cookiecutter.hooks.FailedHookException:
                        pass
                    mock_work_in.assert_called_once_with(repo_dir)
                    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
                    mock_rmtree.assert_not_called()
                    mock_logger.exception.assert_called_once()

def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.side_effect = cookiecutter.exceptions.UndefinedError('Undefined variable')
        with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__.return_value = None
            with unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
                    try:
                        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
                    except cookiecutter.exceptions.UndefinedError:
                        pass
                    mock_work_in.assert_called_once_with(repo_dir)
                    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
                    mock_rmtree.assert_called_once_with(project_dir)
                    mock_logger.exception.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import Mock, patch, mock_open
    
    mock_script_path = '/fake/script.py.j2'
    mock_cwd = '/fake/cwd'
    mock_context = {'cookiecutter': {'project_name': 'TestProject'}}
    mock_template_content = 'print("Hello {{ cookiecutter.project_name }}")'
    mock_rendered_content = 'print("Hello TestProject")'
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env:
        
        mock_temp = Mock()
        mock_temp.name = '/fake/temp.py'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        mock_env = Mock()
        mock_template = Mock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = mock_rendered_content
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_read.assert_called_once_with(encoding='utf-8')
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.j2')
        mock_create_env.assert_called_once_with(mock_context)
        mock_env.from_string.assert_called_once_with(mock_template_content)
        mock_template.render.assert_called_once_with(**mock_context)
        mock_temp.write.assert_called_once_with(mock_rendered_content.encode('utf-8'))
        mock_run_script.assert_called_once_with('/fake/temp.py', mock_cwd)


# LLM-generated content at query #9
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
    assert len(result) == 1
    assert os.path.basename(result[0]) == 'pre_gen_project.py'

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
    hook_file2 = hooks_dir.join('pre_gen_project.sh')
    hook_file2.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert len(result) == 2
    assert sorted([os.path.basename(p) for p in result]) == ['pre_gen_project.py', 'pre_gen_project.sh']

def test_find_hook_with_mixed_files(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    valid_hook = hooks_dir.join('pre_gen_project.py')
    valid_hook.write('')
    backup_hook = hooks_dir.join('pre_gen_project.py~')
    backup_hook.write('')
    unsupported_hook = hooks_dir.join('unsupported.py')
    unsupported_hook.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert len(result) == 1
    assert os.path.basename(result[0]) == 'pre_gen_project.py'


# LLM-generated content at query #10
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

def test_run_pre_prompt_hook_valid_pre_prompt_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.py'
        script_path.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

def test_run_pre_prompt_hook_invalid_hook_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.txt'
        script_path.write_text('test')
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_backup_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.py~'
        script_path.write_text('import sys\nsys.exit(0)')
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_script_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.py'
        script_path.write_text('import sys\nsys.exit(1)')
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_shell_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt'
        script_path.write_text('#!/bin/sh\nexit 0')
        os.chmod(script_path, 0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

def test_run_pre_prompt_hook_empty_script_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt'
        script_path.write_text('')
        os.chmod(script_path, 0o755)
        try:
            run_pre_prompt_hook(tmpdir)
            assert False
        except FailedHookException:
            assert True

def test_run_pre_prompt_hook_multiple_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('import sys\nsys.exit(0)')
        script2 = hooks_dir / 'pre_prompt.sh'
        script2.write_text('#!/bin/sh\nexit 0')
        os.chmod(script2, 0o755)
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()


# LLM-generated content at query #11
#--------------------------

def test_find_hook_hooks_dir_not_directory():
    import os
    import sys
    sys.modules.pop('cookiecutter', None)
    class MockLogger:
        def debug(self, *args, **kwargs):
            pass
    logger = MockLogger()
    sys.modules['cookiecutter'] = type(sys)('cookiecutter')
    sys.modules['cookiecutter'].logger = logger
    import cookiecutter
    def valid_hook(hook_file, hook_name):
        return False
    sys.modules['cookiecutter'].hooks.valid_hook = valid_hook
    from cookiecutter.hooks import find_hook
    hooks_dir = 'non_existent_dir'
    result = find_hook('some_hook', hooks_dir)
    assert result is None


# LLM-generated content at query #12
#--------------------------

def test_valid_hook_matching_supported_not_backup():
    result = valid_hook("/path/to/hook.py", "hook")
    assert result == True

def test_valid_hook_matching_supported_backup():
    result = valid_hook("/path/to/hook.py~", "hook")
    assert result == False

def test_valid_hook_matching_unsupported_not_backup():
    result = valid_hook("/path/to/unknown.py", "unknown")
    assert result == False

def test_valid_hook_nonmatching_supported_not_backup():
    result = valid_hook("/path/to/other.py", "hook")
    assert result == False

def test_valid_hook_nonmatching_unsupported_not_backup():
    result = valid_hook("/path/to/unknown.py", "hook")
    assert result == False

def test_valid_hook_nonmatching_supported_backup():
    result = valid_hook("/path/to/other.py~", "hook")
    assert result == False

def test_valid_hook_matching_unsupported_backup():
    result = valid_hook("/path/to/unknown.py~", "unknown")
    assert result == False

def test_valid_hook_nonmatching_unsupported_backup():
    result = valid_hook("/path/to/unknown.py~", "hook")
    assert result == False


# LLM-generated content at query #13
#--------------------------

def test_hooks_dir_is_not_directory():
    import os
    import sys
    sys.modules.pop('cookiecutter.hooks', None)
    class MockLogger:
        def debug(self, *args):
            pass
    logger = MockLogger()
    os.path.isdir = lambda x: False
    from cookiecutter.hooks import find_hook
    result = find_hook('pre_gen_project', 'non_existent_hooks')
    assert result is None


# LLM-generated content at query #14
#--------------------------

def test_find_hook_with_valid_hook_in_directory():
    import tempfile, os, shutil
    hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result == [os.path.abspath(hook_file_path)]

def test_find_hook_with_no_hooks_directory():
    import tempfile, shutil
    hooks_dir = tempfile.mkdtemp()
    shutil.rmtree(hooks_dir)
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_empty_hooks_directory():
    import tempfile, shutil
    hooks_dir = tempfile.mkdtemp()
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result is None

def test_find_hook_with_unsupported_hook_name():
    import tempfile, os, shutil
    hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(hooks_dir, 'unsupported_hook.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('unsupported_hook', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    import tempfile, os, shutil
    hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py~')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result is None

def test_find_hook_with_mismatching_hook_name():
    import tempfile, os, shutil
    hooks_dir = tempfile.mkdtemp()
    hook_file_path = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import tempfile, os, shutil
    hooks_dir = tempfile.mkdtemp()
    hook_file1_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    hook_file2_path = os.path.join(hooks_dir, 'post_gen_project.py')
    with open(hook_file1_path, 'w') as f:
        f.write('')
    with open(hook_file2_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project', hooks_dir)
    shutil.rmtree(hooks_dir)
    assert result == [os.path.abspath(hook_file1_path)]

def test_find_hook_with_default_hooks_dir():
    import tempfile, os, shutil
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir)
    hook_file_path = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file_path, 'w') as f:
        f.write('')
    result = find_hook('pre_gen_project')
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    assert result == [os.path.abspath(hook_file_path)]


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    from cookiecutter.hooks import run_pre_prompt_hook
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "test_repo"
        repo_dir.mkdir()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_18_evaluates_to_false():
    import sys
    import subprocess
    import errno
    from pathlib import Path
    from unittest.mock import Mock, patch
    EXIT_SUCCESS = 0
    class FailedHookException(Exception):
        pass
    def run_script(script_path: str, cwd: Path | str = '.') -> None:
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
    mock_proc.wait.return_value = EXIT_SUCCESS
    with patch('subprocess.Popen', return_value=mock_proc):
        run_script('test_script.py')
    assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path('/tmp/test_repo')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #18
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

def test_valid_hook_not_matching_unsupported_not_backup():
    result = valid_hook("/some/path/unknown.py", "hook_name")
    assert result == False

def test_valid_hook_matching_supported_backup_with_tilde():
    result = valid_hook("/some/path/hook_name.py~", "hook_name")
    assert result == False

def test_valid_hook_with_different_extension():
    result = valid_hook("/some/path/hook_name.txt", "hook_name")
    assert result == True

def test_valid_hook_empty_hook_name():
    result = valid_hook("/some/path/.py", "")
    assert result == False


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_when_flag_true():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test"}}
    delete_project_on_failure = True
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in, \
         unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         unittest.mock.patch('cookiecutter.hooks.logger') as mock_logger:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #20
#--------------------------

def test_no_hook_found_when_scripts_is_empty():
    scripts = []
    result = not scripts
    assert result is True


# LLM-generated content at query #21
#--------------------------

def test_no_hook_found():
    scripts = []
    result = not scripts
    assert result == True


# LLM-generated content at query #22
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
    assert any(f.suffix == ".py" for f in temp_files)
    
    os.remove(test_script_path)
    for temp_file in temp_files:
        if temp_file.name.startswith("tmp"):
            os.remove(temp_file)


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
        with patch('os.listdir', return_value=[]):
            with patch('os.path.isdir', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_invalid_hook_file_backup():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=['pre_gen_project.py~']):
            with patch('os.path.isdir', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_invalid_hook_file_wrong_name():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=['post_gen_project.py']):
            with patch('os.path.isdir', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1_path = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2_path = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook_file1_path, 'w') as f1:
            f1.write('')
        with open(hook_file2_path, 'w') as f2:
            f2.write('')
        with patch('os.listdir', return_value=['pre_gen_project.py', 'pre_gen_project.sh']):
            with patch('os.path.isdir', return_value=True):
                result = find_hook('pre_gen_project', hooks_dir)
    expected = sorted([os.path.abspath(hook_file1_path), os.path.abspath(hook_file2_path)])
    assert sorted(result) == expected

def test_find_hook_with_unsupported_hook_name():
    import os
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        with patch('os.listdir', return_value=['unsupported_hook.py']):
            with patch('os.path.isdir', return_value=True):
                result = find_hook('unsupported_hook', hooks_dir)
    assert result is None


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

def test_run_pre_prompt_hook_no_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_empty_hooks_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_no_pre_prompt_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        other_hook = hooks_dir / 'post_gen_project.sh'
        other_hook.write_text('#!/bin/bash\necho "test"')
        other_hook.chmod(0o755)
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_valid_pre_prompt_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        pre_prompt_script = hooks_dir / 'pre_prompt.py'
        pre_prompt_script.write_text('print("pre_prompt hook executed")')
        pre_prompt_script.chmod(0o755)
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

def test_run_pre_prompt_hook_valid_pre_prompt_script_with_shebang():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        pre_prompt_script = hooks_dir / 'pre_prompt.sh'
        pre_prompt_script.write_text('#!/bin/bash\necho "pre_prompt hook executed"')
        pre_prompt_script.chmod(0o755)
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

def test_run_pre_prompt_hook_backup_file_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        backup_script = hooks_dir / 'pre_prompt.py~'
        backup_script.write_text('print("backup file")')
        backup_script.chmod(0o755)
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

def test_run_pre_prompt_hook_script_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        pre_prompt_script = hooks_dir / 'pre_prompt.py'
        pre_prompt_script.write_text('import sys\nsys.exit(1)')
        pre_prompt_script.chmod(0o755)
        try:
            hooks.run_pre_prompt_hook(tmpdir)
            assert False
        except hooks.FailedHookException:
            pass

def test_run_pre_prompt_hook_empty_script_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        pre_prompt_script = hooks_dir / 'pre_prompt.sh'
        pre_prompt_script.write_text('')
        pre_prompt_script.chmod(0o755)
        try:
            hooks.run_pre_prompt_hook(tmpdir)
            assert False
        except hooks.FailedHookException:
            pass

def test_run_pre_prompt_hook_multiple_pre_prompt_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('print("script1")')
        script1.chmod(0o755)
        script2 = hooks_dir / 'pre_prompt.sh'
        script2.write_text('#!/bin/bash\necho "script2"')
        script2.chmod(0o755)
        result = hooks.run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()


# LLM-generated content at query #26
#--------------------------

def test_run_pre_prompt_hook_with_valid_pre_prompt_script():
    repo_dir = Path("test_repo")
    scripts = ["pre_prompt.py"]
    find_hook = lambda hook: scripts if hook == "pre_prompt" else None
    run_script = lambda script, repo: None
    with work_in(repo_dir):
        found_scripts = find_hook('pre_prompt')
    assert found_scripts is not None


# LLM-generated content at query #27
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"name": "test"}}
    delete_project_on_failure = True
    with unittest.mock.patch("cookiecutter.hooks.work_in") as mock_work_in, unittest.mock.patch("cookiecutter.hooks.run_hook") as mock_run_hook:
        mock_work_in.return_value.__enter__.return_value = None
        cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_hook_failure_with_deletion():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"name": "test"}}
    delete_project_on_failure = True
    with unittest.mock.patch("cookiecutter.hooks.work_in") as mock_work_in, unittest.mock.patch("cookiecutter.hooks.run_hook") as mock_run_hook, unittest.mock.patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_called_once_with(project_dir)

def test_run_hook_from_repo_dir_hook_failure_without_deletion():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"name": "test"}}
    delete_project_on_failure = False
    with unittest.mock.patch("cookiecutter.hooks.work_in") as mock_work_in, unittest.mock.patch("cookiecutter.hooks.run_hook") as mock_run_hook, unittest.mock.patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"name": "test"}}
    delete_project_on_failure = True
    with unittest.mock.patch("cookiecutter.hooks.work_in") as mock_work_in, unittest.mock.patch("cookiecutter.hooks.run_hook") as mock_run_hook, unittest.mock.patch("cookiecutter.hooks.rmtree") as mock_rmtree:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.exceptions.UndefinedError("Undefined variable")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.exceptions.UndefinedError:
            pass
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #28
#--------------------------

def test_find_hook_returns_none_for_nonexistent_hooks_dir():
    result = find_hook('pre_gen_project', 'nonexistent_dir')
    assert result is None

def test_find_hook_returns_none_for_empty_hooks_dir(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_returns_none_for_no_matching_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hooks_dir.join('post_gen_project.py').write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_returns_none_for_backup_file(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hooks_dir.join('pre_gen_project.py~').write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

def test_find_hook_returns_none_for_unsupported_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hooks_dir.join('unsupported_hook.py').write('')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

def test_find_hook_returns_list_for_single_valid_hook(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project.py')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file)]

def test_find_hook_returns_list_for_multiple_valid_hooks(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file1 = hooks_dir.join('pre_gen_project.py')
    hook_file1.write('')
    hook_file2 = hooks_dir.join('post_gen_project.py')
    hook_file2.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file1)]

def test_find_hook_returns_list_for_valid_hook_without_extension(tmpdir):
    hooks_dir = tmpdir.mkdir('hooks')
    hook_file = hooks_dir.join('pre_gen_project')
    hook_file.write('')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result == [str(hook_file)]


# LLM-generated content at query #29
#--------------------------

def test_run_pre_prompt_hook_without_pre_prompt_script():
    repo_dir = Path('some_template')
    scripts = []
    with mock.patch('cookiecutter.hooks.find_hook', return_value=scripts):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #30
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

def test_run_script_failed_hook_exception():
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
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    finally:
        os.unlink(script_path)

def test_run_script_os_error_enoexec():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_content = ""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o644)
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    except FailedHookException as e:
        assert "might be an empty file or missing a shebang" in str(e)
    else:
        assert False, "Expected FailedHookException"
    finally:
        os.unlink(script_path)

def test_run_script_os_error_other():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_path = "/non/existent/path/script.py"
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)
    else:
        assert False, "Expected FailedHookException"

def test_run_script_cwd_parameter():
    import tempfile
    import os
    import sys
    from pathlib import Path
    script_content = "import sys; sys.exit(0)"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    temp_dir = tempfile.mkdtemp()
    try:
        run_script(script_path, cwd=temp_dir)
    finally:
        os.unlink(script_path)
        os.rmdir(temp_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test Project"}}
    delete_project_on_failure = False
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_delete_project_on_failure_is_false():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False
    with unittest.mock.patch('cookiecutter.hooks.work_in') as mock_work_in, \
         unittest.mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         unittest.mock.patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = cookiecutter.hooks.FailedHookException("Hook failed")
        try:
            cookiecutter.hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        except cookiecutter.hooks.FailedHookException:
            pass
        mock_rmtree.assert_not_called()


# LLM-generated content at query #33
#--------------------------

def test_run_pre_prompt_hook_without_pre_prompt_script():
    repo_dir = Path("some_dir")
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #34
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: None
    original_find_hook = hooks.find_hook
    original_run_script_with_context = hooks.run_script_with_context
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    hooks.find_hook = original_find_hook
    hooks.run_script_with_context = original_run_script_with_context


def test_run_hook_from_repo_dir_hook_not_found():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': None
    original_find_hook = hooks.find_hook
    hooks.find_hook = mock_find_hook
    hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    hooks.find_hook = original_find_hook


def test_run_hook_from_repo_dir_failed_hook_exception_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    mock_rmtree = lambda path: None
    original_find_hook = hooks.find_hook
    original_run_script_with_context = hooks.run_script_with_context
    original_rmtree = utils.rmtree
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass
    hooks.find_hook = original_find_hook
    hooks.run_script_with_context = original_run_script_with_context
    utils.rmtree = original_rmtree


def test_run_hook_from_repo_dir_failed_hook_exception_without_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(hooks.FailedHookException('Hook failed'))
    original_find_hook = hooks.find_hook
    original_run_script_with_context = hooks.run_script_with_context
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except hooks.FailedHookException:
        pass
    hooks.find_hook = original_find_hook
    hooks.run_script_with_context = original_run_script_with_context


def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    mock_find_hook = lambda hook_name, hooks_dir='hooks': ['/tmp/repo/hooks/pre_gen_project.py']
    mock_run_script_with_context = lambda script_path, cwd, context: (_ for _ in ()).throw(jinja2.UndefinedError('Undefined variable'))
    mock_rmtree = lambda path: None
    original_find_hook = hooks.find_hook
    original_run_script_with_context = hooks.run_script_with_context
    original_rmtree = utils.rmtree
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    utils.rmtree = mock_rmtree
    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except jinja2.UndefinedError:
        pass
    hooks.find_hook = original_find_hook
    hooks.run_script_with_context = original_run_script_with_context
    utils.rmtree = original_rmtree


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_rendered_content():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, mock_open, MagicMock
    
    mock_script_path = "/fake/path/script.py"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template_content = "print('{{ cookiecutter.project_name }}')"
    mock_rendered_content = "print('TestProject')"
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.unlink') as mock_unlink:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = mock_rendered_content
        
        mock_temp = MagicMock()
        mock_temp.name = "/fake/temp/file.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_read.assert_called_once_with(encoding='utf-8')
        mock_create_env.assert_called_once_with(mock_context)
        mock_env.from_string.assert_called_once_with(mock_template_content)
        mock_template.render.assert_called_once_with(**mock_context)
        mock_temp.write.assert_called_once_with(mock_rendered_content.encode('utf-8'))
        mock_run_script.assert_called_once_with("/fake/temp/file.py", mock_cwd)
        mock_unlink.assert_called_once_with("/fake/temp/file.py")

def test_run_script_with_context_preserves_file_extension():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, mock_open, MagicMock
    
    mock_script_path = "/fake/path/script.sh"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template_content = "echo {{ cookiecutter.project_name }}"
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.unlink') as mock_unlink:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "echo TestProject"
        
        mock_temp = MagicMock()
        mock_temp.name = "/fake/temp/file.sh"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.sh')

def test_run_script_with_context_handles_path_object():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, mock_open, MagicMock
    
    mock_script_path = Path("/fake/path/script.py")
    mock_cwd = Path("/fake/cwd")
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template_content = "print('{{ cookiecutter.project_name }}')"
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.unlink') as mock_unlink:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "print('TestProject')"
        
        mock_temp = MagicMock()
        mock_temp.name = "/fake/temp/file.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_read.assert_called_once_with(encoding='utf-8')
        mock_run_script.assert_called_once_with("/fake/temp/file.py", Path("/fake/cwd"))

def test_run_script_with_context_passes_context_to_template_render():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, mock_open, MagicMock
    
    mock_script_path = "/fake/path/script.py"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject", "version": "1.0.0"}, "extra": "data"}
    mock_template_content = "print('{{ cookiecutter.project_name }} {{ cookiecutter.version }}')"
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.unlink') as mock_unlink:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = "print('TestProject 1.0.0')"
        
        mock_temp = MagicMock()
        mock_temp.name = "/fake/temp/file.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_template.render.assert_called_once_with(**mock_context)

def test_run_script_with_context_encodes_output_as_utf8():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.hooks import run_script_with_context
    from unittest.mock import patch, mock_open, MagicMock
    
    mock_script_path = "/fake/path/script.py"
    mock_cwd = "/fake/cwd"
    mock_context = {"cookiecutter": {"project_name": "TestProject"}}
    mock_template_content = "print('{{ cookiecutter.project_name }}')"
    mock_rendered_content = "print('TestProject')"
    
    with patch('pathlib.Path.read_text', return_value=mock_template_content) as mock_read, \
         patch('cookiecutter.utils.create_env_with_context') as mock_create_env, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('os.unlink') as mock_unlink:
        
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_create_env.return_value = mock_env
        mock_env.from_string.return_value = mock_template
        mock_template.render.return_value = mock_rendered_content
        
        mock_temp = MagicMock()
        mock_temp.name = "/fake/temp/file.py"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        run_script_with_context(mock_script_path, mock_cwd, mock_context)
        
        mock_temp.write.assert_called_once_with(mock_rendered_content.encode('utf-8'))


# LLM-generated content at query #36
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure_when_enabled():
    mock_repo_dir = "/fake/repo"
    mock_project_dir = "/fake/project"
    mock_context = {"cookiecutter": {"name": "test"}}
    mock_hook_name = "pre_gen_project"
    
    class MockFailedHookException(Exception):
        pass
    
    class MockUndefinedError(Exception):
        pass
    
    original_work_in = cookiecutter.utils.work_in
    original_run_hook = cookiecutter.hooks.run_hook
    original_rmtree = cookiecutter.utils.rmtree
    
    work_in_called_with = None
    run_hook_called_with = None
    rmtree_called_with = None
    logger_exception_called = False
    
    def mock_work_in(dirname):
        work_in_called_with = dirname
        class MockContextManager:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        return MockContextManager()
    
    def mock_run_hook(hook_name, project_dir, context):
        run_hook_called_with = (hook_name, project_dir, context)
        raise MockFailedHookException("Hook failed")
    
    def mock_rmtree(path):
        rmtree_called_with = path
    
    cookiecutter.utils.work_in = mock_work_in
    cookiecutter.hooks.run_hook = mock_run_hook
    cookiecutter.utils.rmtree = mock_rmtree
    
    try:
        cookiecutter.hooks.run_hook_from_repo_dir(
            repo_dir=mock_repo_dir,
            hook_name=mock_hook_name,
            project_dir=mock_project_dir,
            context=mock_context,
            delete_project_on_failure=True
        )
    except MockFailedHookException:
        pass
    
    cookiecutter.utils.work_in = original_work_in
    cookiecutter.hooks.run_hook = original_run_hook
    cookiecutter.utils.rmtree = original_rmtree
    
    assert work_in_called_with == mock_repo_dir
    assert run_hook_called_with == (mock_hook_name, mock_project_dir, mock_context)
    assert rmtree_called_with == mock_project_dir


# LLM-generated content at query #37
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_on_failure_when_flag_false():
    repo_dir = "/tmp/test_repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"cookiecutter": {"project_name": "Test"}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except Exception:
        pass
    assert os.path.exists(project_dir) is True


# LLM-generated content at query #38
#--------------------------

def test_predicate_at_line_21_evaluates_to_false():
    import sys
    import subprocess
    import errno
    from pathlib import Path
    from unittest.mock import Mock, patch
    from pre_commit.errors import FailedHookException
    from pre_commit.util import make_executable
    EXIT_SUCCESS = 0
    original_make_executable = make_executable
    mock_proc = Mock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
        with patch('pre_commit.util.make_executable', side_effect=original_make_executable):
            script_path = '/tmp/test_script.py'
            cwd = Path('.')
            run_thru_shell = sys.platform.startswith('win')
            script_command = [sys.executable, script_path]
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
            predicate_result = False
    assert predicate_result == False


# LLM-generated content at query #39
#--------------------------

```python
def test_run_script_with_context_creates_non_deletable_temp_file():
    script_path = "/tmp/test_script.py"
    cwd = "/tmp"
    context = {"cookiecutter": {"project_name": "test"}}
    temp_file_name = None
    mock_temp_file = type('MockTempFile', (), {'name': '/tmp/temp123.py', 'write': lambda self, data: None})()
    original_NamedTemporaryFile = tempfile.NamedTemporaryFile
    tempfile.NamedTemporaryFile = lambda delete=False, mode='wb', suffix=None: mock_temp_file
    original_create_env_with_context = utils.create_env_with_context
    mock_env = type('MockEnv', (), {'from_string': lambda self, contents: type('MockTemplate', (), {'render': lambda self, **kwargs: 'rendered content'})()})()
    utils.create_env_with_context = lambda context: mock_env
    original_run_script = hooks.run_script
    hooks.run_script = lambda script, cwd: None
    hooks.run_script_with_context(script_path, cwd, context)
    tempfile.NamedTemporaryFile = original_NamedTemporaryFile
    utils.create_env_with_context = original_create_env_with_context
    hooks.run_script = original_run_script
    assert tempfile.NamedTemporaryFile(delete=False).delete == False


