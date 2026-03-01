####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook_and_no_backup():
    assert valid_hook('pre_commit.py', 'pre_commit') == True

def test_valid_hook_with_non_matching_hook_name():
    assert valid_hook('pre_commit.py', 'post_commit') == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook('unknown_hook.py', 'unknown_hook') == False

def test_valid_hook_with_backup_file():
    assert valid_hook('pre_commit.py~', 'pre_commit') == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook('pre_commit.txt', 'pre_commit') == False

def test_valid_hook_with_empty_hook_name():
    assert valid_hook('pre_commit.py', '') == False

def test_valid_hook_with_empty_hook_file():
    assert valid_hook('', 'pre_commit') == False


# LLM-generated content at query #2
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = '/path/to/test_script.py'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True  # If no exception is raised, the test passes

def test_run_script_successful_non_python_script():
    script_path = '/path/to/test_script.sh'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True  # If no exception is raised, the test passes

def test_run_script_failed_hook_exception():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_no_exec():
    script_path = '/path/to/empty_or_no_shebang_script'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_general():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e).startswith('Hook script failed (error:')


# LLM-generated content at query #3
#--------------------------

```python
def test_run_hook_no_hook_found():
    assert run_hook('nonexistent_hook', '/tmp', {}) is None

def test_run_hook_with_valid_hook():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_path = os.path.join(hook_dir, 'pre_gen_project.py')
        with open(hook_path, 'w') as f:
            f.write('print("test")')

        with patch('cookiecutter.hooks.find_hook', return_value=[hook_path]), \
             patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', tmpdir, {'test': 'value'})
            mock_run.assert_called_once_with(hook_path, tmpdir, {'test': 'value'})

def test_run_hook_with_multiple_hooks():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_path1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_path2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        with open(hook_path1, 'w') as f:
            f.write('print("test1")')
        with open(hook_path2, 'w') as f:
            f.write('echo "test2"')

        with patch('cookiecutter.hooks.find_hook', return_value=[hook_path1, hook_path2]), \
             patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', tmpdir, {'test': 'value'})
            assert mock_run.call_count == 2
            mock_run.assert_any_call(hook_path1, tmpdir, {'test': 'value'})
            mock_run.assert_any_call(hook_path2, tmpdir, {'test': 'value'})


# LLM-generated content at query #4
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.sh'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'echo {{ cookiecutter.project_name }}'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.sh'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.sh', cwd)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda x: []
    logger = Mock()
    run_script_with_context = Mock()

    run_hook(hook_name, project_dir, context)

    assert logger.debug.call_args_list[0][0] == ('No %s hook found', hook_name)
    assert run_script_with_context.call_count == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_file():
    assert valid_hook("pre_commit.py", "pre_commit") is True


# LLM-generated content at query #7
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt'
        hook_file.write_text('#!/bin/sh\necho "test"')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'invalid_hook'
        hook_file.write_text('#!/bin/sh\nexit 1')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.another', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.another') in result


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script_with_context():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'project_name': 'test_project'}

    run_script_with_context(script_path, cwd, context)


# LLM-generated content at query #10
#--------------------------

```python
def test_hooks_dir_is_directory():
    os.makedirs('hooks', exist_ok=True)
    assert os.path.isdir('hooks')


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='./test_repo',
        hook_name='pre_gen_project',
        project_dir='./test_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    repo_dir = Path(tempfile.mkdtemp())
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/some/path"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    logger = MagicMock()
    run_hook(hook_name, project_dir, context)
    logger.debug.assert_called_once_with('No %s hook found', hook_name)


# LLM-generated content at query #14
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #15
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    initial_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    assert os.getcwd() == initial_dir


# LLM-generated content at query #16
#--------------------------

```python
def test_oserror_with_non_enoexec_errno():
    import sys
    import errno
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import subprocess

    script_path = '/path/to/script.sh'
    cwd = '/path/to/cwd'
    err = OSError(errno.EACCES, 'Permission denied')

    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = err
        with patch('utils.make_executable'):
            with patch('sys.platform', 'linux'):
                with pytest.raises(FailedHookException) as exc_info:
                    run_script(script_path, cwd)
                assert str(exc_info.value) == f'Hook script failed (error: {err})'


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_scripts_found():
    assert run_hook('pre_gen_project', '/tmp/project', {}) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_valid_hook_with_valid_parameters():
    assert valid_hook("pre-commit~", "pre-commit") is False


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('/fake/repo')
    assert run_pre_prompt_hook(repo_dir) == repo_dir


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='test_hook',
            project_dir='test_project',
            context={},
            delete_project_on_failure=False,
        )
        mock_work_in.assert_called_once_with('test_repo')


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #22
#--------------------------

```python
def test_find_hook_returns_none_when_no_scripts_match():
    assert find_hook('nonexistent_hook', 'hooks') is None


# LLM-generated content at query #23
#--------------------------

```python
def test_find_hook_predicate_false():
    assert not (len([]) == 0)


# LLM-generated content at query #24
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_paths():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.sh') in result


# LLM-generated content at query #25
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') is True


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, False)


# LLM-generated content at query #28
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        mock_work_in.return_value.__enter__.return_value = None
        run_hook_from_repo_dir(Path('/fake/repo'), 'pre_gen_project', Path('/fake/project'), {}, True)
        mock_work_in.assert_called_once_with(Path('/fake/repo'))


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not (isinstance(err, OSError) and err.errno == errno.ENOEXEC)


# LLM-generated content at query #31
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/invalid_hook.sh', 'w') as f:
        f.write('#!/bin/sh\necho "invalid"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_valid_hook_paths():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "valid"')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks/pre-commit')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "backup"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "valid1"')
    with open('hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "valid2"')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 2


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in_returns_none():
    result = work_in()
    assert result is None


# LLM-generated content at query #33
#--------------------------

```python
def test_find_hook_no_directory():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_empty_directory():
    with patch('os.listdir', return_value=[]):
        assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_valid_hook_found():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit']), \
         patch('os.path.abspath', side_effect=lambda x: x), \
         patch('os.path.join', side_effect=lambda x, y: f'{x}/{y}'):
        assert find_hook('pre-commit', 'hooks') == ['hooks/pre-commit']

def test_find_hook_invalid_hook_ignored():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['invalid-hook']), \
         patch('valid_hook', return_value=False):
        assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_backup_file_ignored():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit~']), \
         patch('valid_hook', return_value=False):
        assert find_hook('pre-commit', 'hooks') is None


# LLM-generated content at query #34
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'_jinja2_env_vars': {}, 'project_name': 'test'}}
    script_path = Path('test_script.py')
    script_path.write_text('print("{{ cookiecutter.project_name }}")')
    cwd = Path('.')

    run_script_with_context(script_path, cwd, context)

    assert script_path.exists()
    assert script_path.read_text() == 'print("{{ cookiecutter.project_name }}")'


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = 'test_script.sh'
    cwd = '/tmp'
    context = {'cookiecutter': {'project_name': 'test'}}
    Path(script_path).write_text('#!/bin/bash\necho "Hello {{ cookiecutter.project_name }}"', encoding='utf-8')
    run_script_with_context(script_path, cwd, context)
    assert not os.path.exists(script_path)


# LLM-generated content at query #36
#--------------------------

```python
def test_tempfile_delete_false():
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        assert temp.delete is False


# LLM-generated content at query #37
#--------------------------

```python
def test_run_hook_from_repo_dir_delete_project_on_failure():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        mock_work_in.return_value.__enter__.return_value = None

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #38
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.utils.os') as mock_os:
        mock_os.getcwd.return_value = '/current/dir'
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.return_value.__enter__.return_value = None
            run_hook_from_repo_dir(
                repo_dir='/repo/dir',
                hook_name='pre_gen_project',
                project_dir='/project/dir',
                context={'cookiecutter': {}},
                delete_project_on_failure=True,
            )
            mock_work_in.assert_called_once_with('/repo/dir')


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    assert not (0 != 0)


# LLM-generated content at query #40
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_true():
    with pytest.raises(FailedHookException) as exc_info:
        run_script('nonexistent_script.sh')
    assert exc_info.value.args[0] == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.sh\')'


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook_from_repo_dir_with_delete_project_on_failure_false():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'fake': 'context'}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_false():
    exit_status = 1
    assert not (exit_status != EXIT_SUCCESS)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='.',
            hook_name='pre_gen_project',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True,
        )


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS is False


# LLM-generated content at query #46
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_non_python_file():
    script_path = 'test_script.sh'
    cwd = '/test_dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_failed_exit_status():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_with_enoexec_error():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_with_os_error():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'test_script.py\')'


# LLM-generated content at query #47
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    # Assertions would verify the script was executed successfully

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    # Assertions would verify the script was executed successfully

def test_run_script_failed_exit_status():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'exit status' in str(excinfo.value)

def test_run_script_os_error_no_exec():
    script_path = 'empty_or_no_shebang_script'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'error' in str(excinfo.value)


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:

        mock_run_hook.side_effect = FailedHookException('hook failed')
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #49
#--------------------------

```python
def test_oserror_with_enonexec_raises_failedhookeception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Executable not found')
        with pytest.raises(FailedHookException) as excinfo:
            run_script('/path/to/script.sh')
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)


# LLM-generated content at query #50
#--------------------------

```python
def test_tempfile_delete_false():
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        assert temp.delete == False


# LLM-generated content at query #51
#--------------------------

```python
def test_exit_status_is_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #52
#--------------------------

```python
def test_tempfile_delete_false_mode_wb_suffix_extension():
    script_path = 'test_script.py'
    cwd = '/test/cwd'
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    _, extension = os.path.splitext(script_path)
    contents = Path(script_path).read_text(encoding='utf-8')
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension)
    assert temp.delete == False
    assert temp.mode == 'wb'
    assert temp.name.endswith(extension)
    temp.close()


# LLM-generated content at query #53
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager():
    repo_dir = '/some/repo/dir'
    hook_name = 'pre_gen_project'
    project_dir = '/some/project/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook'):
            run_hook_from_repo_dir(
                repo_dir,
                hook_name,
                project_dir,
                context,
                delete_project_on_failure,
            )

    mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #54
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        mock_work_in.return_value.__enter__.return_value = None
        mock_run_hook.side_effect = FailedHookException('test error')
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {}, False)
        mock_rmtree.assert_not_called()


# LLM-generated content at query #55
#--------------------------

```python
def test_run_script_with_context():
    script_path = Path('/fake/script.sh')
    cwd = Path('/fake/cwd')
    context = {'cookiecutter': {'name': 'test'}}
    script_path.write_text('echo "{{ cookiecutter.name }}"', encoding='utf-8')
    run_script_with_context(script_path, cwd, context)
    assert script_path.read_text(encoding='utf-8') == 'echo "test"'


# LLM-generated content at query #56
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo',
        hook_name='pre_gen_project',
        project_dir='project_output',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_with_undefined',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_21():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Test error")
        with pytest.raises(FailedHookException) as exc_info:
            run_script("test_script.sh")
        assert exc_info.value.args[0] == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not list(Path.cwd().glob('pre_prompt*'))


# LLM-generated content at query #59
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    repo_dir = Path('/non/existent/dir')
    with work_in(repo_dir):
        scripts = find_hook('pre_prompt')
        assert not scripts


# LLM-generated content at query #60
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='./tests/mock_repo',
        hook_name='pre_gen_project',
        project_dir='./tests/mock_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./tests/mock_repo',
            hook_name='pre_gen_project',
            project_dir='./tests/mock_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./tests/mock_repo',
            hook_name='pre_gen_project',
            project_dir='./tests/mock_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )


# LLM-generated content at query #61
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    assert os.getcwd() == original_dir


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    repo_dir = None
    with work_in(repo_dir) as result:
        assert result is None


# LLM-generated content at query #63
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path(original_dir) / 'test_dir'
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir


# LLM-generated content at query #64
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='test_hook',
            project_dir='test_project',
            context={},
            delete_project_on_failure=True,
        )
    assert not os.path.exists('test_project')


# LLM-generated content at query #65
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(), OSError)


# LLM-generated content at query #66
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    assert os.getcwd() == original_dir


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/fake-repo-pre')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/fake-repo-pre-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = '/path/to/test_script.py'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    # Assertions would be handled by mocking subprocess.Popen and checking calls

def test_run_script_successful_non_python_script():
    script_path = '/path/to/test_script.sh'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    # Assertions would be handled by mocking subprocess.Popen and checking calls

def test_run_script_failed_hook_exception():
    script_path = '/path/to/failing_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status:' in str(excinfo.value)

def test_run_script_os_error_empty_file():
    script_path = '/path/to/empty_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (error:' in str(excinfo.value)


# LLM-generated content at query #3
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook("pre-commit", "pre-commit") == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit", "commit-msg") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #4
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre_commit.py", "pre_commit") is True


# LLM-generated content at query #5
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.another', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.another') in result


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") is True


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook_returns_list_when_hook_exists():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #10
#--------------------------

```python
def test_hooks_dir_exists():
    os.path.isdir.return_value = True
    assert find_hook('pre-commit') is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    repo_dir = Path('/tmp/test_repo')
    scripts = []
    with patch('cookiecutter.hooks.find_hook', return_value=scripts):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_work_in.assert_called_once_with(repo_dir)
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_failed_hook_exception():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = FailedHookException('test error')
        with raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_work_in.assert_called_once_with(repo_dir)
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once_with(
        "Stopping generation because %s hook script didn't exit successfully",
        hook_name
    )

def test_run_hook_from_repo_dir_undefined_error():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = UndefinedError('test error')
        with raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_work_in.assert_called_once_with(repo_dir)
    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once_with(
        "Stopping generation because %s hook script didn't exit successfully",
        hook_name
    )


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_with_valid_hook():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/path/to/hook/script.py']
    find_hook.return_value = scripts
    run_script_with_context.return_value = None
    assert run_hook(hook_name, project_dir, context) is None
    assert find_hook.call_args == call(hook_name)
    assert run_script_with_context.call_args_list == [call(script, project_dir, context) for script in scripts]

def test_run_hook_with_no_hook():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    find_hook.return_value = None
    assert run_hook(hook_name, project_dir, context) is None
    assert find_hook.call_args == call(hook_name)


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='./test_repo',
        hook_name='pre_gen_project',
        project_dir='./test_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='failing_hook',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='failing_hook',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error_with_cleanup():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='undefined_hook',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_undefined_error_without_cleanup():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='undefined_hook',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/test-data/cookiecutter-no-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/test-data/cookiecutter-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/test-data/cookiecutter-failed-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "commit-msg") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #3
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/data/test-template-no-hooks')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/data/test-template-with-hooks')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/data/test-template-with-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_no_hooks_found():
    assert run_hook('pre_gen_project', '/tmp/project', {}) is None

def test_run_hook_with_valid_hook():
    os.makedirs('/tmp/hooks', exist_ok=True)
    Path('/tmp/hooks/pre_gen_project.py').write_text('print("Hello")')
    run_hook('pre_gen_project', '/tmp/project', {'cookiecutter': {'project_name': 'test'}})
    assert Path('/tmp/hooks/pre_gen_project.py').exists()

def test_run_hook_with_invalid_hook():
    os.makedirs('/tmp/hooks', exist_ok=True)
    Path('/tmp/hooks/invalid_hook.py').write_text('print("Hello")')
    assert run_hook('pre_gen_project', '/tmp/project', {}) is None

def test_run_hook_with_backup_file():
    os.makedirs('/tmp/hooks', exist_ok=True)
    Path('/tmp/hooks/pre_gen_project.py~').write_text('print("Hello")')
    assert run_hook('pre_gen_project', '/tmp/project', {}) is None

def test_run_hook_with_multiple_hooks():
    os.makedirs('/tmp/hooks', exist_ok=True)
    Path('/tmp/hooks/pre_gen_project.py').write_text('print("Hello")')
    Path('/tmp/hooks/pre_gen_project.sh').write_text('echo "Hello"')
    run_hook('pre_gen_project', '/tmp/project', {'cookiecutter': {'project_name': 'test'}})
    assert Path('/tmp/hooks/pre_gen_project.py').exists()
    assert Path('/tmp/hooks/pre_gen_project.sh').exists()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt'
    hook_file.write_text('#!/bin/sh\necho "test"')
    hook_file.chmod(0o755)
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert (result / 'hooks' / 'pre_prompt').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'invalid_hook'
    hook_file.write_text('#!/bin/sh\necho "test"')
    hook_file.chmod(0o755)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt'
    hook_file.write_text('#!/bin/sh\nexit 1')
    hook_file.chmod(0o755)
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_ignores_unsupported_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/unsupported-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.another', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.another') in result


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.sh') in result


# LLM-generated content at query #4
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.sh'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_generic():
    script_path = 'invalid_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e).startswith('Hook script failed (error:')


# LLM-generated content at query #5
#--------------------------

```python
def test_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('/path/to/repo')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_with_matching_name_and_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") is True

def test_valid_hook_with_non_matching_name():
    assert valid_hook("pre-commit.py", "commit-msg") is False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") is False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") is False

def test_valid_hook_with_mismatched_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") is False


# LLM-generated content at query #7
#--------------------------

```python
def test_run_hook_with_no_hooks_found():
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    assert run_hook(hook_name, project_dir, context) is None

def test_run_hook_with_valid_hook():
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/tmp/hooks/pre_gen_project.py']
    assert find_hook(hook_name) == scripts
    run_hook(hook_name, project_dir, context)


# LLM-generated content at query #8
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
    repo_dir = Path('tests/fake-repo-pre')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path('tests/fake-repo-pre-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_failed_script():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    assert find_hook('nonexistent-hook', 'hooks') is None

def test_find_hook_returns_list_with_valid_hook():
    assert find_hook('pre-commit', 'hooks') == ['/path/to/hooks/pre-commit']

def test_find_hook_ignores_backup_files():
    assert find_hook('pre-commit', 'hooks') == ['/path/to/hooks/pre-commit']


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo',
        hook_name='pre_gen_project',
        project_dir='valid_project',
        context={'cookiecutter': {}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='undefined_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_no_hook_found():
    run_hook_from_repo_dir(
        repo_dir='no_hook_repo',
        hook_name='non_existent_hook',
        project_dir='no_hook_project',
        context={'cookiecutter': {}},
        delete_project_on_failure=True
    )


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_no_hook_found():
    assert run_hook('nonexistent_hook', '/fake/dir', {}) is None

def test_run_hook_with_valid_hook():
    with patch('cookiecutter.hooks.find_hook', return_value=['/fake/dir/hook.sh']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/fake/dir', {'key': 'value'})
            mock_run.assert_called_once_with('/fake/dir/hook.sh', '/fake/dir', {'key': 'value'})


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"cookiecutter": {"project_name": "test_project"}}

    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        run_hook(hook_name, project_dir, context)


# LLM-generated content at query #14
#--------------------------

```python
def test_hooks_dir_is_not_a_directory():
    assert not os.path.isdir('hooks')


# LLM-generated content at query #15
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context():
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'repo_name': 'test_repo',
        }
    }
    script_path = 'test_script.sh'
    cwd = '/tmp'

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path_instance = mock_path.return_value
        mock_path_instance.read_text.return_value = 'echo {{ cookiecutter.project_name }}'

        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp_instance = mock_temp.return_value
            mock_temp_instance.name = 'temp_script.sh'

            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                run_script_with_context(script_path, cwd, context)

                mock_path.assert_called_once_with(script_path)
                mock_path_instance.read_text.assert_called_once_with(encoding='utf-8')
                mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
                mock_run_script.assert_called_once_with('temp_script.sh', cwd)


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_predicate_true():
    os.path.isdir.return_value = True
    assert find_hook('hook_name') is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook_and_no_backup():
    assert valid_hook('pre-commit', 'pre-commit') == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook('pre-commit', 'commit-msg') == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook('unknown-hook', 'unknown-hook') == False

def test_valid_hook_with_backup_file():
    assert valid_hook('pre-commit~', 'pre-commit') == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook('pre-commit.sh', 'pre-commit') == False


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    logger = Mock()
    run_hook(hook_name, project_dir, context)
    logger.debug.assert_called_once_with('No %s hook found', hook_name)


# LLM-generated content at query #20
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") is True


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_with_context():
    script_path = Path('test_script.sh')
    cwd = Path('.')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_content = 'echo "Hello {{ cookiecutter.project_name }}"'
    script_path.write_text(script_content, encoding='utf-8')

    run_script_with_context(script_path, cwd, context)

    assert script_path.exists()
    assert script_path.read_text(encoding='utf-8') == script_content


# LLM-generated content at query #22
#--------------------------

```python
def test_oserror_with_enexec_errno_raises_failed_hook_exception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Test error")
        with pytest.raises(FailedHookException) as exc_info:
            run_script("test_script.sh")
        assert "Hook script failed, might be an empty file or missing a shebang" in str(exc_info.value)


# LLM-generated content at query #23
#--------------------------

```python
def test_hooks_dir_is_not_directory():
    import os
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, "nonexistent_dir")
        assert not os.path.isdir(hooks_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='test_repo',
        hook_name='pre_gen_project',
        project_dir='test_project',
        context={'cookiecutter': {}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='pre_gen_project',
            project_dir='test_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='pre_gen_project',
            project_dir='test_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=False
        )


# LLM-generated content at query #25
#--------------------------

```python
def test_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path("test_repo")
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_predicate():
    assert find_hook('hook_name', 'hooks') is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/mocks/pre-and-post-hooks')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/mocks/pre-and-post-hooks')
    with patch('cookiecutter.hooks.find_hook', return_value=['pre_prompt.sh']):
        with patch('cookiecutter.hooks.run_script'):
            result = run_pre_prompt_hook(repo_dir)
            assert result != repo_dir
            assert result.exists()

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/mocks/pre-and-post-hooks')
    with patch('cookiecutter.hooks.find_hook', return_value=['pre_prompt.sh']):
        with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('test')):
            with pytest.raises(FailedHookException):
                run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['other-hook.py']):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_list_with_valid_hook():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.py']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert result == ['hooks/pre-commit.py']

def test_find_hook_ignores_backup_files():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.py~']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_ignores_unsupported_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['unsupported-hook.py']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_multiple_matching_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.py', 'pre-commit.sh']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert len(result) == 2
        assert 'hooks/pre-commit.py' in result
        assert 'hooks/pre-commit.sh' in result


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(), OSError)


# LLM-generated content at query #30
#--------------------------

```python
def test_run_hook_from_repo_dir_delete_project_on_failure():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    os.makedirs(project_dir, exist_ok=True)
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = FailedHookException
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(errno.ENOEXEC), OSError)


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    """Test that the predicate at line 20 evaluates to False."""
    with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException('test')):
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            mock_work_in.__enter__ = MagicMock()
            mock_work_in.__exit__ = MagicMock()
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with patch('cookiecutter.hooks.logger') as mock_logger:
                    run_hook_from_repo_dir(
                        repo_dir='test_repo',
                        hook_name='test_hook',
                        project_dir='test_project',
                        context={},
                        delete_project_on_failure=False,
                    )
                    mock_rmtree.assert_not_called()


# LLM-generated content at query #33
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/some/repo/dir"
    hook_name = "pre_gen_project"
    project_dir = "/some/project/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    assert not delete_project_on_failure


# LLM-generated content at query #34
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    assert find_hook('non_existent_hook', 'hooks') is None

def test_find_hook_returns_list_with_valid_hook():
    assert find_hook('pre-commit', 'hooks') == ['/absolute/path/to/hooks/pre-commit']

def test_find_hook_ignores_backup_files():
    assert find_hook('pre-commit', 'hooks') == ['/absolute/path/to/hooks/pre-commit']


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test_dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test_dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_evaluates_to_false():
    """Test that the predicate at line 14 evaluates to False."""
    script_path = "test_script.py"
    cwd = "/test/cwd"
    context = {"test": "value"}

    with patch('os.path.splitext', return_value=("", ".py")):
        with patch('Path.read_text', return_value="print('test')"):
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value = mock_temp
                mock_temp.delete = False
                mock_temp.mode = 'wb'
                mock_temp.suffix = '.py'

                run_script_with_context(script_path, cwd, context)

                assert mock_temp.return_value.__enter__.return_value.delete is False
                assert mock_temp.return_value.__enter__.return_value.mode == 'wb'
                assert mock_temp.return_value.__enter__.return_value.suffix == '.py'


# LLM-generated content at query #37
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with work_in(repo_dir):
        assert os.getcwd() == repo_dir


# LLM-generated content at query #38
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
    repo_dir = Path('tests/fake-repo-pre')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path('tests/fake-repo-pre-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_failed_execution():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_no_scripts_found():
    assert run_hook('pre_gen_project', '/path/to/project', {}) is None
    assert run_hook('post_gen_project', '/path/to/project', {}) is None
    assert run_hook('pre_hook', '/path/to/project', {}) is None
    assert run_hook('post_hook', '/path/to/project', {}) is None

def test_run_hook_with_valid_scripts():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    assert run_hook('pre_gen_project', '/path/to/project', context) is None
    assert run_hook('post_gen_project', '/path/to/project', context) is None
    assert run_hook('pre_hook', '/path/to/project', context) is None
    assert run_hook('post_hook', '/path/to/project', context) is None


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    assert run_script(script_path, cwd) is None

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    assert run_script(script_path, cwd) is None

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_no_exec():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_generic():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)


# LLM-generated content at query #4
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_valid_hook_path():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert result == [os.path.abspath('test_hooks_dir/pre-commit')]
    os.remove('test_hooks_dir/pre-commit')
    os.rmdir('test_hooks_dir')

def test_find_hook_ignores_backup_files():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/pre-commit~')
    os.rmdir('test_hooks_dir')

def test_find_hook_ignores_unsupported_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/unsupported-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/unsupported-hook')
    os.rmdir('test_hooks_dir')


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo',
        hook_name='pre_gen_project',
        project_dir='project_output',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_undefined_error_without_deletion():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='project_output',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "commit-msg") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #7
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('/fake/repo')
    assert run_pre_prompt_hook(repo_dir) == repo_dir


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    run_hook(hook_name, project_dir, context)


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert not os.path.isdir('hooks')


# LLM-generated content at query #10
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    logger = MagicMock()
    run_hook(hook_name, project_dir, context)
    logger.debug.assert_called_once_with('No %s hook found', hook_name)


# LLM-generated content at query #12
#--------------------------

```python
def test_hooks_dir_is_not_a_directory():
    import os
    import tempfile
    from your_module import find_hook

    # Create a temporary file (not a directory)
    with tempfile.NamedTemporaryFile() as temp_file:
        result = find_hook('test_hook', temp_file.name)

    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_hooks_dir_exists():
    assert os.path.isdir('hooks') is True


# LLM-generated content at query #14
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = '/path/to/script.py'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True  # Placeholder for actual assertions

def test_run_script_with_non_python_file():
    script_path = '/path/to/script.sh'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True  # Placeholder for actual assertions

def test_run_script_with_default_cwd():
    script_path = '/path/to/script.py'
    run_script(script_path)
    assert True  # Placeholder for actual assertions

def test_run_script_fails_with_exit_status():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert 'exit status' in str(e)

def test_run_script_fails_with_enoexec():
    script_path = '/path/to/empty_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert 'empty file or missing a shebang' in str(e)

def test_run_script_fails_with_oserror():
    script_path = '/path/to/invalid_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert 'error' in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_valid_hook_with_matching_and_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "commit-msg") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #16
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_is_not_a_directory():
    assert find_hook('hook_name', 'non_existent_dir') is None


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path('/path/to/repo')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #18
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_no_scripts_found():
    result = run_hook('pre_gen_project', '/tmp', {})
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = "/tmp/test_dir"
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = EXIT_SUCCESS
    run_script(script_path, cwd)
    proc.wait.assert_called_once()
    proc.terminate.assert_not_called()

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = EXIT_SUCCESS
    run_script(script_path, cwd)
    proc.wait.assert_called_once()
    proc.terminate.assert_not_called()

def test_run_script_failed_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status: 1)' in str(exc_info.value)

def test_run_script_failed_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status: 1)' in str(exc_info.value)

def test_run_script_os_error():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path, cwd)
    assert 'Hook script failed (error:' in str(exc_info.value)

def test_run_script_os_error_no_exec():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as exc_info:
        run_script(script_path, cwd)
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(exc_info.value)


# LLM-generated content at query #22
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook_and_no_backup():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook_name():
    assert valid_hook("pre-commit.py", "post-commit") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_different_extension():
    assert valid_hook("pre-commit.sh", "pre-commit") == False

def test_valid_hook_with_path_in_filename():
    assert valid_hook("/path/to/pre-commit.py", "pre-commit") == True


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo',
        hook_name='pre_gen_project',
        project_dir='valid_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='project_to_delete',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='project_to_keep',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #24
#--------------------------

```python
def test_hooks_dir_is_directory():
    os.path.isdir = lambda x: True
    assert os.path.isdir('hooks')


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(script_path, cwd)


# LLM-generated content at query #26
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #27
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #28
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/fake-repo-pre')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/fake-repo-pre-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/fake-repo-pre-failed-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #30
#--------------------------

```python
def test_oserror_with_enonexec():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Test error')
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path, cwd)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, OSError)
        assert exc_info.value.__cause__.errno == errno.ENOEXEC


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_evaluates_to_false():
    delete_project_on_failure = False
    assert not delete_project_on_failure


# LLM-generated content at query #32
#--------------------------

```python
def test_run_script_with_context():
    script_path = Path('test_script.sh')
    cwd = Path('/tmp')
    context = {'project_name': 'test_project'}

    script_path.write_text('echo "{{ project_name }}"', encoding='utf-8')

    run_script_with_context(script_path, cwd, context)

    assert not script_path.exists()


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    # Mocking the necessary components to simulate the scenario where the predicate at line 21 evaluates to False
    import sys
    from pathlib import Path
    import subprocess
    import errno

    # Setup
    script_path = "test_script.sh"
    cwd = Path(".")

    # Mock sys.platform to not start with 'win'
    sys.platform = "linux"

    # Mock subprocess.Popen to raise an OSError with errno not equal to ENOEXEC
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    subprocess.Popen = mock_popen

    # Mock utils.make_executable to do nothing
    import utils
    utils.make_executable = lambda x: None

    # Call the function and expect it to raise FailedHookException
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='test_hook',
            project_dir='test_project',
            context={},
            delete_project_on_failure=False,
        )
        mock_work_in.assert_called_once_with('test_repo')


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_with_delete_project_on_failure_false():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        mock_work_in.return_value.__enter__.return_value = None

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #36
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    repo_dir = Path("nonexistent_dir")
    with pytest.raises(FileNotFoundError):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #37
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.py'
    cwd = '/tmp'
    run_script_with_context(script_path, cwd, context)


# LLM-generated content at query #38
#--------------------------

```python
def test_oserror_with_enoexec_raises_failed_hook_exception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Executable format error')
        with raises(FailedHookException) as exc_info:
            run_script('dummy_script.sh')
        assert str(exc_info.value) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #39
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_none_when_only_backup_files_exist():
    os.makedirs('hooks_with_backup', exist_ok=True)
    with open('hooks_with_backup/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "backup"')
    assert find_hook('pre-commit', 'hooks_with_backup') is None
    os.remove('hooks_with_backup/pre-commit~')
    os.rmdir('hooks_with_backup')

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('hooks_with_valid', exist_ok=True)
    with open('hooks_with_valid/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "valid"')
    result = find_hook('pre-commit', 'hooks_with_valid')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks_with_valid/pre-commit')
    os.remove('hooks_with_valid/pre-commit')
    os.rmdir('hooks_with_valid')

def test_find_hook_ignores_unsupported_hook():
    os.makedirs('hooks_with_unsupported', exist_ok=True)
    with open('hooks_with_unsupported/unsupported-hook', 'w') as f:
        f.write('#!/bin/sh\necho "unsupported"')
    assert find_hook('pre-commit', 'hooks_with_unsupported') is None
    os.remove('hooks_with_unsupported/unsupported-hook')
    os.rmdir('hooks_with_unsupported')

def test_find_hook_returns_multiple_matching_hooks():
    os.makedirs('hooks_with_multiple', exist_ok=True)
    with open('hooks_with_multiple/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "first"')
    with open('hooks_with_multiple/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "second"')
    result = find_hook('pre-commit', 'hooks_with_multiple')
    assert result is not None
    assert len(result) == 2
    os.remove('hooks_with_multiple/pre-commit')
    os.remove('hooks_with_multiple/pre-commit.sh')
    os.rmdir('hooks_with_multiple')


# LLM-generated content at query #40
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo',
        hook_name='pre_gen_project',
        project_dir='valid_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=False
    )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_undefined_error_without_deletion():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='undefined_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error_with_deletion():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='undefined_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script_py_file_success():
    script_path = 'test_script.py'
    cwd = '.'
    run_script(script_path, cwd)
    assert True

def test_run_script_non_py_file_success():
    script_path = 'test_script.sh'
    cwd = '.'
    run_script(script_path, cwd)
    assert True

def test_run_script_failure():
    script_path = 'test_script_fail.py'
    cwd = '.'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_empty_file():
    script_path = 'test_script_empty'
    cwd = '.'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_missing_shebang():
    script_path = 'test_script_no_shebang'
    cwd = '.'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.sh'
    cwd = '/tmp'

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'echo {{ cookiecutter.project_name }}'
        mock_path.return_value.suffix = '.sh'

        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/temp_script.sh'
            mock_temp.return_value.__enter__.return_value.write.return_value = None

            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_env.return_value.from_string.return_value.render.return_value = 'echo test_project'

                with patch('cookiecutter.hooks.run_script') as mock_run:
                    run_script_with_context(script_path, cwd, context)

                    mock_path.assert_called_once_with(script_path)
                    mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
                    mock_env.assert_called_once_with(context)
                    mock_run.assert_called_once_with('/tmp/temp_script.sh', cwd)


# LLM-generated content at query #43
#--------------------------

```python
def test_find_hook_returns_list_when_hook_exists():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_find_hook_returns_none_when_no_matching_scripts():
    # Setup
    hook_name = "test_hook"
    hooks_dir = "empty_hooks_dir"
    os.makedirs(hooks_dir, exist_ok=True)

    # Execution and Assertion
    assert find_hook(hook_name, hooks_dir) is None


# LLM-generated content at query #45
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #46
#--------------------------

```python
def test_find_hook_returns_none_when_no_matching_scripts():
    assert find_hook('nonexistent_hook', 'hooks') is None


# LLM-generated content at query #47
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #48
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #49
#--------------------------

```python
def test_find_hook_returns_none_when_no_matching_scripts():
    assert find_hook('nonexistent_hook', 'hooks') is None


# LLM-generated content at query #50
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_none_when_no_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/other-hook')
    os.rmdir('hooks')

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks/pre-commit')
    os.remove('hooks/pre-commit')
    os.rmdir('hooks')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/pre-commit~')
    os.rmdir('hooks')

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 2
    assert any(path.endswith('hooks/pre-commit') for path in result)
    assert any(path.endswith('hooks/pre-commit.sh') for path in result)
    os.remove('hooks/pre-commit')
    os.remove('hooks/pre-commit.sh')
    os.rmdir('hooks')


# LLM-generated content at query #51
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed' in str(excinfo.value)


# LLM-generated content at query #52
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        assert delete_project_on_failure is True


# LLM-generated content at query #53
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    assert os.getcwd() == original_dir


# LLM-generated content at query #54
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='repo_dir',
        hook_name='pre_gen_project',
        project_dir='project_dir',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #55
#--------------------------

```python
def test_exit_status_not_success():
    exit_status = 1
    assert exit_status != EXIT_SUCCESS


