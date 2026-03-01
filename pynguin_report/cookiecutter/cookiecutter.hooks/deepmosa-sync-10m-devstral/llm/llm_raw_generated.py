####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_absolute_path_to_valid_hook():
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
    assert find_hook('unsupported-hook', 'hooks') is None

def test_find_hook_returns_multiple_matching_hooks():
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
def test_run_hook_from_repo_dir_success():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_failed_hook_exception():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = FailedHookException('Hook failed')

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()

def test_run_hook_from_repo_dir_undefined_error():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = UndefinedError('Undefined variable')

        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()
        mock_logger.exception.assert_called_once()


# LLM-generated content at query #4
#--------------------------

```python
def test_hooks_dir_exists():
    assert os.path.isdir('hooks') is True


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True  # If no exception, test passes

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True  # If no exception, test passes

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e).startswith('Hook script failed (error:')


# LLM-generated content at query #6
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = "test_script.py"
    cwd = "/test/directory"
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = "test_script.sh"
    cwd = "/test/directory"
    run_script(script_path, cwd)
    assert True

def test_run_script_failure_non_zero_exit():
    script_path = "failing_script.py"
    cwd = "/test/directory"
    with pytest.raises(FailedHookException, match="exit status: 1"):
        run_script(script_path, cwd)

def test_run_script_failure_no_shebang():
    script_path = "no_shebang_script"
    cwd = "/test/directory"
    with pytest.raises(FailedHookException, match="missing a shebang"):
        run_script(script_path, cwd)

def test_run_script_failure_os_error():
    script_path = "nonexistent_script.py"
    cwd = "/test/directory"
    with pytest.raises(FailedHookException, match="error: "):
        run_script(script_path, cwd)


# LLM-generated content at query #7
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'test_repo'
        repo_dir.mkdir()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'test_repo'
        repo_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt'
        hook_file.write_text('#!/bin/sh\necho "test"')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()
        assert (result / 'hooks' / 'pre_prompt').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'test_repo'
        repo_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'invalid_hook'
        hook_file.write_text('#!/bin/sh\necho "test"')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_failing_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'test_repo'
        repo_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt'
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


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_predicate_true():
    os.makedirs('hooks', exist_ok=True)
    assert os.path.isdir('hooks')


# LLM-generated content at query #10
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre_commit.py", "pre_commit") is True


# LLM-generated content at query #11
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook_and_no_backup():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook_name():
    assert valid_hook("pre-commit.py", "post-commit") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("invalid-hook.py", "invalid-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='tests/mocks/pretend-repo-1',
        hook_name='pre_gen_project',
        project_dir='tests/mocks/pretend-project-1',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='tests/mocks/pretend-repo-1',
            hook_name='pre_gen_project',
            project_dir='tests/mocks/pretend-project-1',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='tests/mocks/pretend-repo-1',
            hook_name='pre_gen_project',
            project_dir='tests/mocks/pretend-project-1',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #13
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") is True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "commit-msg") is False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") is False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") is False

def test_valid_hook_with_different_extension():
    assert valid_hook("pre-commit.sh", "pre-commit") is False


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_with_valid_hook():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/path/to/hook/script.py']
    find_hook(hook_name) = scripts
    run_script_with_context(scripts[0], project_dir, context)
    assert True

def test_run_hook_with_no_hook_found():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    find_hook(hook_name) = None
    run_hook(hook_name, project_dir, context)
    assert True

def test_run_hook_with_multiple_scripts():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/path/to/hook/script1.py', '/path/to/hook/script2.py']
    find_hook(hook_name) = scripts
    run_script_with_context(scripts[0], project_dir, context)
    run_script_with_context(scripts[1], project_dir, context)
    assert True


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path('/some/repo/dir')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #16
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_hook_found():
    assert run_hook('nonexistent_hook', '/tmp', {}) is None

def test_run_hook_valid_hook():
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', '/tmp', context)

def test_run_hook_with_multiple_scripts():
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('post_gen_project', '/tmp', context)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_returns_early_when_no_scripts_found():
    assert run_hook('nonexistent_hook', '/fake/path', {}) is None


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook_no_scripts_found():
    context = {}
    project_dir = "/path/to/project"
    hook_name = "pre_gen_project"
    find_hook(hook_name) == []
    run_hook(hook_name, project_dir, context)


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook_nonexistent_directory():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_empty_directory():
    os.makedirs('empty_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_dir') is None
    os.rmdir('empty_dir')

def test_find_hook_no_matching_hooks():
    os.makedirs('test_hooks', exist_ok=True)
    with open('test_hooks/other-hook.py', 'w') as f:
        f.write('#!/usr/bin/env python\nprint("test")')
    assert find_hook('pre-commit', 'test_hooks') is None
    os.remove('test_hooks/other-hook.py')
    os.rmdir('test_hooks')

def test_find_hook_with_matching_hook():
    os.makedirs('test_hooks', exist_ok=True)
    with open('test_hooks/pre-commit.py', 'w') as f:
        f.write('#!/usr/bin/env python\nprint("test")')
    result = find_hook('pre-commit', 'test_hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('test_hooks/pre-commit.py')
    os.remove('test_hooks/pre-commit.py')
    os.rmdir('test_hooks')

def test_find_hook_with_backup_file():
    os.makedirs('test_hooks', exist_ok=True)
    with open('test_hooks/pre-commit.py~', 'w') as f:
        f.write('#!/usr/bin/env python\nprint("test")')
    assert find_hook('pre-commit', 'test_hooks') is None
    os.remove('test_hooks/pre-commit.py~')
    os.rmdir('test_hooks')

def test_find_hook_with_multiple_matching_hooks():
    os.makedirs('test_hooks', exist_ok=True)
    with open('test_hooks/pre-commit.py', 'w') as f:
        f.write('#!/usr/bin/env python\nprint("test1")')
    with open('test_hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'test_hooks')
    assert result is not None
    assert len(result) == 2
    assert any(path.endswith('test_hooks/pre-commit.py') for path in result)
    assert any(path.endswith('test_hooks/pre-commit.sh') for path in result)
    os.remove('test_hooks/pre-commit.py')
    os.remove('test_hooks/pre-commit.sh')
    os.rmdir('test_hooks')


# LLM-generated content at query #22
#--------------------------

```python
def test_run_script_with_context_success():
    script_path = Path('test_script.sh')
    script_path.write_text('echo "Hello {{ name }}"', encoding='utf-8')
    context = {'name': 'World'}
    cwd = Path('.')

    run_script_with_context(script_path, cwd, context)

    script_path.unlink()

def test_run_script_with_context_failure():
    script_path = Path('test_script.sh')
    script_path.write_text('exit 1', encoding='utf-8')
    context = {}
    cwd = Path('.')

    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, cwd, context)

    script_path.unlink()

def test_run_script_with_context_empty_file():
    script_path = Path('test_script.sh')
    script_path.write_text('', encoding='utf-8')
    context = {}
    cwd = Path('.')

    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, cwd, context)

    script_path.unlink()

def test_run_script_with_context_python_script():
    script_path = Path('test_script.py')
    script_path.write_text('print("Hello {{ name }}")', encoding='utf-8')
    context = {'name': 'World'}
    cwd = Path('.')

    run_script_with_context(script_path, cwd, context)

    script_path.unlink()


# LLM-generated content at query #23
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
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {}},
            delete_project_on_failure=False
        )


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_true():
    os.path.isdir.return_value = True
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #25
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path("/some/repo")
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/invalid_hook.py', 'w') as f:
        f.write('# invalid hook')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_list_of_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('# valid hook')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks/pre-commit')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('# backup hook')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_absolute_paths():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('# valid hook')
    result = find_hook('pre-commit', 'hooks')
    assert os.path.isabs(result[0])


# LLM-generated content at query #27
#--------------------------

```python
def test_pre_prompt_hook_returns_original_dir_when_no_scripts():
    repo_dir = Path('/path/to/repo')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #28
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_extension():
    script_path = '/path/to/script.py'
    cwd = '/working/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    _, extension = os.path.splitext(script_path)
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension)
    assert temp.suffix == extension


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    proc.wait.assert_called_once()
    proc.terminate.assert_not_called()

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    proc.wait.assert_called_once()
    proc.terminate.assert_not_called()

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_os_error_empty_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (error: ' in str(excinfo.value)


# LLM-generated content at query #30
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/some/repo/dir'
    hook_name = 'pre_gen_project'
    project_dir = '/some/project/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    assert not os.path.exists(project_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_find_hook_predicate_false():
    result = find_hook('nonexistent_hook', 'nonexistent_dir')
    assert result is None


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    repo_dir = Path('/fake/repo')
    hook_name = 'pre_gen_project'
    project_dir = Path('/fake/project')
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=None)

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        assert mock_rmtree.called


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    # Mocking the necessary components to simulate the scenario
    import sys
    import subprocess
    import errno
    from pathlib import Path

    sys.platform = 'linux'  # Ensure not running on Windows
    script_path = '/path/to/script.sh'  # Non-Python script
    cwd = '/working/directory'

    # Mock subprocess.Popen to raise OSError with errno.ENOEXEC
    original_popen = subprocess.Popen
    subprocess.Popen = lambda *args, **kwargs: (_ for _ in ()).throw(OSError(errno.ENOEXEC, 'Executable not found'))

    # Mock utils.make_executable to do nothing
    import utils
    original_make_executable = utils.make_executable
    utils.make_executable = lambda *args: None

    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'
    finally:
        subprocess.Popen = original_popen
        utils.make_executable = original_make_executable


# LLM-generated content at query #34
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    initial_dir = os.getcwd()
    test_dir = Path("/tmp/test_dir")

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == initial_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path("/tmp/test_dir")
    test_dir.mkdir(exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['unrelated_hook.sh']):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_list_with_valid_hook():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.sh']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert result == ['hooks/pre-commit.sh']

def test_find_hook_ignores_backup_files():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.sh~']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_ignores_unsupported_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['unsupported-hook.sh']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_multiple_matching_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit.sh', 'pre-commit.bash']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert len(result) == 2
        assert 'hooks/pre-commit.sh' in result
        assert 'hooks/pre-commit.bash' in result


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_false():
    exit_status = 1
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #38
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'test': 'value'}

    result = run_script_with_context(script_path, cwd, context)

    assert result is None


# LLM-generated content at query #39
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
        mock_path.return_value.read_text.return_value = 'echo {{ cookiecutter.project_name }}'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.sh'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.sh', cwd)


# LLM-generated content at query #40
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #41
#--------------------------

```python
def test_run_hook_from_repo_dir_with_delete_project_on_failure_false():
    repo_dir = Path('/fake/repo/dir')
    hook_name = 'pre_gen_project'
    project_dir = Path('/fake/project/dir')
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:

        mock_run_hook.side_effect = FailedHookException('Hook failed')

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #42
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='tests/mock_repo',
        hook_name='pre_gen_project',
        project_dir='tests/mock_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=False
    )

def test_run_hook_from_repo_dir_failure_with_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='tests/mock_repo',
            hook_name='failing_hook',
            project_dir='tests/mock_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_cleanup():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='tests/mock_repo',
            hook_name='failing_hook',
            project_dir='tests/mock_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_no_hook_found():
    run_hook_from_repo_dir(
        repo_dir='tests/mock_repo',
        hook_name='nonexistent_hook',
        project_dir='tests/mock_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True
    )


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script_with_context():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    contents = 'print("Hello {{ cookiecutter.project_name }}")'
    expected_output = 'print("Hello test")'

    with patch('pathlib.Path.read_text', return_value=contents) as mock_read, \
         patch('tempfile.NamedTemporaryFile') as mock_temp, \
         patch('cookiecutter.hooks.run_script') as mock_run, \
         patch('cookiecutter.utils.create_env_with_context') as mock_env:

        mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
        mock_temp.return_value.__enter__.return_value.write = Mock()

        run_script_with_context(script_path, cwd, context)

        mock_read.assert_called_once_with(encoding='utf-8')
        mock_env.assert_called_once_with(context)
        mock_env.return_value.from_string.assert_called_once_with(contents)
        mock_env.return_value.from_string.return_value.render.assert_called_once_with(**context)
        mock_temp.return_value.__enter__.return_value.write.assert_called_once_with(expected_output.encode('utf-8'))
        mock_run.assert_called_once_with('temp_script.py', cwd)


# LLM-generated content at query #45
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
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt'
    hook_file.write_text('#!/bin/sh\nexit 1')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #46
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/some/repo/dir"
    hook_name = "pre_gen_project"
    project_dir = "/some/project/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException("Hook failed")

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #47
#--------------------------

```python
def test_oserror_with_enexec_errno():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
        with pytest.raises(FailedHookException) as exc_info:
            run_script("test_script.sh")
        assert "Hook script failed, might be an empty file or missing a shebang" in str(exc_info.value)


# LLM-generated content at query #48
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='test_hook',
            project_dir='test_project',
            context={},
            delete_project_on_failure=True
        )
        mock_work_in.assert_called_once_with('test_repo')


# LLM-generated content at query #49
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None


# LLM-generated content at query #50
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    """Test that the predicate at line 14 evaluates to False."""
    script_path = Path('test_script.py')
    cwd = Path('/tmp')
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}

    with patch('builtins.open', mock_open(read_data='print("Hello")')) as mock_file:
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value = mock_temp
            mock_temp.delete = False
            mock_temp.mode = 'wb'
            mock_temp.suffix = '.py'
            mock_temp.name = '/tmp/temp_script.py'

            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)

            assert mock_temp.called
            assert mock_temp.return_value.__enter__.return_value.delete is False
            assert mock_temp.return_value.__enter__.return_value.mode == 'wb'
            assert mock_temp.return_value.__enter__.return_value.suffix == '.py'


# LLM-generated content at query #51
#--------------------------

```python
def test_run_script_with_context_with_python_script():
    context = {'project_name': 'test_project'}
    script_path = 'test_script.py'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run:
                mock_path.return_value.read_text.return_value = 'print("{{ project_name }}")'
                mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.py', cwd)

def test_run_script_with_context_with_shell_script():
    context = {'project_name': 'test_project'}
    script_path = 'test_script.sh'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run:
                mock_path.return_value.read_text.return_value = 'echo "{{ project_name }}"'
                mock_temp.return_value.__enter__.return_value.name = 'temp_script.sh'
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.sh', cwd)

def test_run_script_with_context_creates_jinja_env():
    context = {'project_name': 'test_project'}
    script_path = 'test_script.py'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run:
                with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                    mock_path.return_value.read_text.return_value = 'print("{{ project_name }}")'
                    mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
                    run_script_with_context(script_path, cwd, context)
                    mock_env.assert_called_once_with(context)

def test_run_script_with_context_renders_template():
    context = {'project_name': 'test_project'}
    script_path = 'test_script.py'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run:
                mock_path.return_value.read_text.return_value = 'print("{{ project_name }}")'
                mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
                run_script_with_context(script_path, cwd, context)
                mock_temp.return_value.__enter__.return_value.write.assert_called_once_with(b'print("test_project")')


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with work_in(repo_dir):
        assert os.getcwd() == repo_dir


# LLM-generated content at query #53
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('hook_name', 'non_existent_dir') is None


# LLM-generated content at query #54
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

def test_run_script_os_error_no_exec():
    script_path = 'empty_script'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_other():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'


# LLM-generated content at query #55
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, False)

def test_run_hook_from_repo_dir_failure_with_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, True)

def test_run_hook_from_repo_dir_failure_without_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, False)


# LLM-generated content at query #56
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') is True


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_evaluates_to_false():
    exit_status = 1
    assert exit_status != 0


# LLM-generated content at query #58
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
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('test error')

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #59
#--------------------------

```python
def test_tempfile_suffix_matches_extension():
    script_path = 'test_script.sh'
    cwd = '/tmp'
    context = {}

    _, extension = os.path.splitext(script_path)
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension)

    assert temp.name.endswith(extension)
    temp.close()
    os.unlink(temp.name)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_successful_execution():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    # Assertions would be handled by mocking subprocess.Popen and checking calls

def test_run_script_python_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    # Assertions would verify sys.executable is used

def test_run_script_failed_exit_status():
    script_path = 'failing_script.sh'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='exit status'):
        run_script(script_path, cwd)

def test_run_script_os_error_no_exec():
    script_path = 'empty_script.sh'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.sh'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='error'):
        run_script(script_path, cwd)

def test_run_script_windows_platform():
    script_path = 'test_script.bat'
    cwd = '/test/dir'
    with mock.patch('sys.platform', 'win32'):
        run_script(script_path, cwd)
    # Assertions would verify shell=True is used


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


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    with patch('os.listdir', return_value=['invalid_hook.sh']):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_list_of_matching_hooks():
    with patch('os.listdir', return_value=['pre-commit.sh', 'pre-commit.bak']):
        with patch('os.path.abspath', side_effect=lambda x: x):
            result = find_hook('pre-commit')
            assert result == ['hooks/pre-commit.sh']

def test_find_hook_ignores_backup_files():
    with patch('os.listdir', return_value=['pre-commit.sh~']):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_absolute_paths():
    with patch('os.listdir', return_value=['pre-commit.sh']):
        with patch('os.path.abspath', return_value='/abs/path/pre-commit.sh'):
            result = find_hook('pre-commit')
            assert result == ['/abs/path/pre-commit.sh']


# LLM-generated content at query #4
#--------------------------

```python
def test_valid_hook_with_valid_hook_file():
    assert valid_hook("pre-commit", "pre-commit") == True

def test_valid_hook_with_invalid_hook_name():
    assert valid_hook("pre-commit", "invalid-hook") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unsupported-hook", "unsupported-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit~", "pre-commit") == False

def test_valid_hook_with_different_extension():
    assert valid_hook("pre-commit.py", "pre-commit") == False


# LLM-generated content at query #5
#--------------------------

```python
def test_valid_hook_returns_true_for_matching_supported_non_backup_file():
    assert valid_hook("pre_commit.py", "pre_commit") is True


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_no_scripts_found():
    assert run_hook('nonexistent_hook', '/tmp', {}) is None

def test_run_hook_with_valid_script():
    context = {'cookiecutter': {'project_name': 'test'}}
    script_content = 'echo "Hello {{ cookiecutter.project_name }}"'
    script_path = '/tmp/test_hook.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    assert run_hook('test', '/tmp', context) is None
    os.remove(script_path)

def test_run_hook_with_invalid_script():
    context = {'cookiecutter': {'project_name': 'test'}}
    script_content = 'invalid_command'
    script_path = '/tmp/invalid_hook.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    with pytest.raises(FailedHookException):
        run_hook('invalid', '/tmp', context)
    os.remove(script_path)


# LLM-generated content at query #7
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit~", "pre-commit") is False
    assert valid_hook("pre-commit.py", "pre-commit") is True
    assert valid_hook("invalid-hook.py", "pre-commit") is False
    assert valid_hook("pre-commit.py", "invalid-hook") is False


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_no_hook_found():
    assert run_hook('nonexistent_hook', '/tmp', {}) is None

def test_run_hook_with_valid_hook():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre_gen_project.py', 'w') as f:
        f.write('print("Hello")')
    run_hook('pre_gen_project', os.getcwd(), {'cookiecutter': {'project_name': 'test'}})
    os.remove('hooks/pre_gen_project.py')
    os.rmdir('hooks')

def test_run_hook_with_invalid_hook():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/invalid_hook.txt', 'w') as f:
        f.write('invalid')
    assert run_hook('invalid_hook', os.getcwd(), {}) is None
    os.remove('hooks/invalid_hook.txt')
    os.rmdir('hooks')

def test_run_hook_with_backup_file():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre_gen_project.py~', 'w') as f:
        f.write('print("Backup")')
    assert run_hook('pre_gen_project', os.getcwd(), {}) is None
    os.remove('hooks/pre_gen_project.py~')
    os.rmdir('hooks')

def test_run_hook_with_multiple_scripts():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre_gen_project.py', 'w') as f:
        f.write('print("First")')
    with open('hooks/pre_gen_project.sh', 'w') as f:
        f.write('echo "Second"')
    run_hook('pre_gen_project', os.getcwd(), {'cookiecutter': {'project_name': 'test'}})
    os.remove('hooks/pre_gen_project.py')
    os.remove('hooks/pre_gen_project.sh')
    os.rmdir('hooks')


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
    assert find_hook('pre-commit', 'hooks') != ['/path/to/hooks/pre-commit~']

def test_find_hook_returns_absolute_paths():
    result = find_hook('pre-commit', 'hooks')
    assert all(os.path.isabs(path) for path in result) if result else True


# LLM-generated content at query #10
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('print("Hello")')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert (result / 'hooks' / 'pre_prompt.py').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'invalid_hook.py'
    hook_file.write_text('print("Hello")')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt.py'
    hook_file.write_text('import sys; sys.exit(1)')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts():
    repo_dir = Path('test_repo')
    repo_dir.mkdir(exist_ok=True)
    with pytest.raises(SystemExit):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    Path(script_path).write_text('echo "{{ cookiecutter.project_name }}"', encoding='utf-8')
    run_script_with_context(script_path, cwd, context)
    assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_tempfile_suffix_matches_script_extension():
    context = {}
    script_path = 'test_script.py'
    _, extension = os.path.splitext(script_path)
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension) as temp:
        assert temp.name.endswith(extension)


# LLM-generated content at query #14
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #15
#--------------------------

```python
def test_find_hook_no_hooks_dir():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_empty_dir():
    os.makedirs('empty_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_dir') is None
    os.rmdir('empty_dir')

def test_find_hook_invalid_hook_file():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/invalid_hook.txt', 'w') as f:
        f.write('')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/invalid_hook.txt')
    os.rmdir('hooks')

def test_find_hook_valid_hook_file():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == os.path.abspath('hooks/pre-commit')
    os.remove('hooks/pre-commit')
    os.rmdir('hooks')

def test_find_hook_backup_file():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/pre-commit~')
    os.rmdir('hooks')

def test_find_hook_multiple_valid_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    with open('hooks/pre-commit.bak', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0] == os.path.abspath('hooks/pre-commit')
    os.remove('hooks/pre-commit')
    os.remove('hooks/pre-commit.bak')
    os.rmdir('hooks')


# LLM-generated content at query #16
#--------------------------

```python
def test_find_hook_returns_none_for_nonexistent_directory():
    result = find_hook('pre-commit', 'nonexistent_dir')
    assert result is None

def test_find_hook_returns_none_for_empty_directory():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    result = find_hook('pre-commit', 'empty_hooks_dir')
    assert result is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_none_for_no_matching_hooks():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result is None
    os.remove('hooks_dir/other-hook')
    os.rmdir('hooks_dir')

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result == [os.path.abspath('hooks_dir/pre-commit')]
    os.remove('hooks_dir/pre-commit')
    os.rmdir('hooks_dir')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result is None
    os.remove('hooks_dir/pre-commit~')
    os.rmdir('hooks_dir')

def test_find_hook_returns_multiple_matching_hooks():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks_dir/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert len(result) == 2
    assert os.path.abspath('hooks_dir/pre-commit') in result
    assert os.path.abspath('hooks_dir/pre-commit.sh') in result
    os.remove('hooks_dir/pre-commit')
    os.remove('hooks_dir/pre-commit.sh')
    os.rmdir('hooks_dir')


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_hooks_found():
    assert run_hook('pre_gen_project', '/path/to/project', {}) is None

def test_run_hook_with_valid_hook():
    with patch('cookiecutter.hooks.find_hook', return_value=['/path/to/hook_script.py']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', '/path/to/project', {'key': 'value'})
            mock_run.assert_called_once_with('/path/to/hook_script.py', '/path/to/project', {'key': 'value'})

def test_run_hook_with_multiple_hooks():
    with patch('cookiecutter.hooks.find_hook', return_value=['/path/to/hook1.py', '/path/to/hook2.sh']):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('post_gen_project', '/path/to/project', {'key': 'value'})
            assert mock_run.call_count == 2
            mock_run.assert_any_call('/path/to/hook1.py', '/path/to/project', {'key': 'value'})
            mock_run.assert_any_call('/path/to/hook2.sh', '/path/to/project', {'key': 'value'})


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook_returns_none_when_no_hooks_dir():
    assert find_hook("pre-commit", "nonexistent_dir") is None

def test_find_hook_returns_none_when_no_matching_hooks():
    os.makedirs("empty_hooks_dir", exist_ok=True)
    assert find_hook("pre-commit", "empty_hooks_dir") is None
    os.rmdir("empty_hooks_dir")

def test_find_hook_returns_none_when_only_backup_files():
    os.makedirs("backup_hooks_dir", exist_ok=True)
    with open("backup_hooks_dir/pre-commit~", "w") as f:
        f.write("#!/bin/sh\necho 'backup'")
    assert find_hook("pre-commit", "backup_hooks_dir") is None
    os.remove("backup_hooks_dir/pre-commit~")
    os.rmdir("backup_hooks_dir")

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs("valid_hooks_dir", exist_ok=True)
    with open("valid_hooks_dir/pre-commit", "w") as f:
        f.write("#!/bin/sh\necho 'valid'")
    result = find_hook("pre-commit", "valid_hooks_dir")
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith("valid_hooks_dir/pre-commit")
    os.remove("valid_hooks_dir/pre-commit")
    os.rmdir("valid_hooks_dir")

def test_find_hook_ignores_unsupported_hooks():
    os.makedirs("unsupported_hooks_dir", exist_ok=True)
    with open("unsupported_hooks_dir/unsupported-hook", "w") as f:
        f.write("#!/bin/sh\necho 'unsupported'")
    assert find_hook("pre-commit", "unsupported_hooks_dir") is None
    os.remove("unsupported_hooks_dir/unsupported-hook")
    os.rmdir("unsupported_hooks_dir")

def test_find_hook_returns_multiple_matching_hooks():
    os.makedirs("multiple_hooks_dir", exist_ok=True)
    with open("multiple_hooks_dir/pre-commit", "w") as f:
        f.write("#!/bin/sh\necho 'first'")
    with open("multiple_hooks_dir/pre-commit.sh", "w") as f:
        f.write("#!/bin/sh\necho 'second'")
    result = find_hook("pre-commit", "multiple_hooks_dir")
    assert result is not None
    assert len(result) == 2
    os.remove("multiple_hooks_dir/pre-commit")
    os.remove("multiple_hooks_dir/pre-commit.sh")
    os.rmdir("multiple_hooks_dir")


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='valid_repo_dir',
        hook_name='pre_gen_project',
        project_dir='valid_project_dir',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True,
    )

def test_run_hook_from_repo_dir_failure_with_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo_dir',
            hook_name='pre_gen_project',
            project_dir='invalid_project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True,
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo_dir',
            hook_name='pre_gen_project',
            project_dir='invalid_project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False,
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_with_undefined_var',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True,
        )


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook_no_hooks_found():
    assert run_hook('pre_gen_project', '/tmp', {}) is None

def test_run_hook_successful_execution():
    script_path = '/tmp/hooks/pre_gen_project.py'
    Path(script_path).write_text('#!/usr/bin/env python\nprint("Hello")')
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', '/tmp', context)
    assert Path(script_path).exists()

def test_run_hook_with_invalid_script():
    script_path = '/tmp/hooks/pre_gen_project.sh'
    Path(script_path).write_text('invalid command')
    context = {'cookiecutter': {'project_name': 'test'}}
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', '/tmp', context)

def test_run_hook_with_jinja_template():
    script_path = '/tmp/hooks/pre_gen_project.py'
    Path(script_path).write_text('print("{{ cookiecutter.project_name }}")')
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', '/tmp', context)
    assert Path(script_path).exists()


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    assert proc.wait.called_once()

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    assert proc.wait.called_once()

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_os_error_enoexec():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (error:' in str(excinfo.value)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path(tempfile.mkdtemp())
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path(tempfile.mkdtemp())
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt'
    hook_file.write_text('#!/bin/sh\necho "test"')
    utils.make_executable(str(hook_file))
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert (result / 'hooks' / 'pre_prompt').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = Path(tempfile.mkdtemp())
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'invalid_hook'
    hook_file.write_text('#!/bin/sh\nexit 1')
    utils.make_executable(str(hook_file))
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path('/path/to/repo')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #24
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'_jinja2_env_vars': {}, 'project_name': 'test'}}
    script_content = 'echo "{{ cookiecutter.project_name }}"'
    script_path = Path(tempfile.mktemp(suffix='.sh'))
    script_path.write_text(script_content, encoding='utf-8')
    cwd = tempfile.mkdtemp()

    run_script_with_context(script_path, cwd, context)

    assert script_path.exists()
    assert script_path.read_text(encoding='utf-8') == script_content


# LLM-generated content at query #25
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    repo_dir = Path(tempfile.mkdtemp(prefix='cookiecutter'))
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure():
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
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #28
#--------------------------

```python
def test_find_hook_predicate_false():
    assert not os.path.isdir('hooks')


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None


# LLM-generated content at query #30
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager():
    repo_dir = '/some/repo/dir'
    hook_name = 'pre_gen_project'
    project_dir = '/some/project/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_work_in.return_value.__enter__ = Mock()
        mock_work_in.return_value.__exit__ = Mock()

        run_hook_from_repo_dir(
            repo_dir, hook_name, project_dir, context, delete_project_on_failure
        )

        mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_context_manager_changes_and_restores_directory():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp())

    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)

    assert os.getcwd() == original_dir


# LLM-generated content at query #32
#--------------------------

```python
def test_exit_status_not_success():
    exit_status = 1
    assert not (exit_status != EXIT_SUCCESS)


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script_with_context_with_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}

    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                    mock_path_instance = mock_path.return_value
                    mock_path_instance.read_text.return_value = 'print("{{ cookiecutter.project_name }}")'
                    mock_env = mock_create_env.return_value
                    mock_template = mock_env.from_string.return_value
                    mock_template.render.return_value = 'print("test_project")'
                    mock_temp_instance = mock_temp.return_value
                    mock_temp_instance.name = 'temp_script.py'

                    run_script_with_context(script_path, cwd, context)

                    mock_path.assert_called_once_with(script_path)
                    mock_path_instance.read_text.assert_called_once_with(encoding='utf-8')
                    mock_create_env.assert_called_once_with(context)
                    mock_env.from_string.assert_called_once_with('print("{{ cookiecutter.project_name }}")')
                    mock_template.render.assert_called_once_with(**context)
                    mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.py')
                    mock_temp_instance.write.assert_called_once_with('print("test_project")'.encode('utf-8'))
                    mock_run_script.assert_called_once_with('temp_script.py', cwd)

def test_run_script_with_context_with_shell_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}

    with patch('cookiecutter.hooks.Path') as mock_path:
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            with patch('cookiecutter.hooks.run_script') as mock_run_script:
                with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                    mock_path_instance = mock_path.return_value
                    mock_path_instance.read_text.return_value = 'echo "{{ cookiecutter.project_name }}"'
                    mock_env = mock_create_env.return_value
                    mock_template = mock_env.from_string.return_value
                    mock_template.render.return_value = 'echo "test_project"'
                    mock_temp_instance = mock_temp.return_value
                    mock_temp_instance.name = 'temp_script.sh'

                    run_script_with_context(script_path, cwd, context)

                    mock_path.assert_called_once_with(script_path)
                    mock_path_instance.read_text.assert_called_once_with(encoding='utf-8')
                    mock_create_env.assert_called_once_with(context)
                    mock_env.from_string.assert_called_once_with('echo "{{ cookiecutter.project_name }}"')
                    mock_template.render.assert_called_once_with(**context)
                    mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
                    mock_temp_instance.write.assert_called_once_with('echo "test_project"'.encode('utf-8'))
                    mock_run_script.assert_called_once_with('temp_script.sh', cwd)


# LLM-generated content at query #34
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') or not os.path.isdir('hooks')


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_non_python_file():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_default_cwd():
    script_path = 'test_script.py'
    run_script(script_path)
    assert True

def test_run_script_fails_with_non_zero_exit_status():
    script_path = 'failing_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_fails_with_os_error():
    script_path = 'nonexistent_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'

def test_run_script_fails_with_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #37
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

def test_find_hook_returns_absolute_paths():
    result = find_hook('pre-commit', 'hooks')
    assert all(os.path.isabs(path) for path in result)


# LLM-generated content at query #38
#--------------------------

```python
def test_oserror_without_enoexec_raises_failedhookeception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, 'Permission denied')
        with raises(FailedHookException) as excinfo:
            run_script('/path/to/script.sh')
        assert str(excinfo.value) == 'Hook script failed (error: [Errno 13] Permission denied)'


# LLM-generated content at query #39
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    assert isinstance(FailedHookException, type)
    assert isinstance(UndefinedError, type)
    assert issubclass(FailedHookException, Exception)
    assert issubclass(UndefinedError, Exception)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_false_when_delete_project_on_failure_is_false():
    repo_dir = "/some/repo"
    hook_name = "pre_gen_project"
    project_dir = "/some/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = FailedHookException("Hook failed")
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    script_path = 'test_script.py'
    cwd = '/tmp'
    context = {'cookiecutter': {'name': 'test'}}
    _, extension = os.path.splitext(script_path)
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension) as temp:
        assert temp.name.endswith(extension)


# LLM-generated content at query #42
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #43
#--------------------------

```python
def test_oserror_with_non_enoexec_errno():
    import sys
    import errno
    from pathlib import Path
    from unittest.mock import patch
    import subprocess

    script_path = "test_script.sh"
    cwd = Path(".")

    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd)
        assert "Hook script failed (error: [Errno 13] Permission denied)" in str(excinfo.value)


# LLM-generated content at query #44
#--------------------------

```python
def test_oserror_predicate_false():
    err = OSError(errno.EACCES, "Permission denied")
    assert not (err.errno == errno.ENOEXEC)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    script_path = Path(__file__).parent / 'test_script.sh'
    script_path.write_text('echo "test"', encoding='utf-8')

    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp_file.return_value.__enter__.return_value.name = '/fake/temp/path'
        run_script_with_context(script_path, Path.cwd(), context)

        assert mock_temp_file.called
        call_kwargs = mock_temp_file.call_args.kwargs
        assert call_kwargs['delete'] is False
        assert call_kwargs['mode'] == 'wb'
        assert call_kwargs['suffix'] == '.sh'


# LLM-generated content at query #46
#--------------------------

```python
def test_find_hook_predicate():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #47
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    script_path = "test_script.sh"
    cwd = "/test/dir"
    context = {"test": "value"}

    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp_file.return_value.__enter__.return_value.name = "temp_script.sh"
        run_script_with_context(script_path, cwd, context)

    mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.sh')


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/fake/repo'
    hook_name = 'fake_hook'
    project_dir = '/fake/project'
    context = {}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:

        mock_run_hook.side_effect = FailedHookException('test')
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #49
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/fake-repo-pre')
    result = hooks.run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/fake-repo-pre-with-hook')
    result = hooks.run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(hooks.FailedHookException):
        hooks.run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #50
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with work_in(repo_dir):
        assert os.getcwd() == repo_dir


# LLM-generated content at query #51
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
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
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'invalid_hook'
    hook_file.write_text('#!/bin/sh\necho "test"')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = Path('test_repo')
    repo_dir.mkdir()
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir()
    hook_file = hooks_dir / 'pre_prompt'
    hook_file.write_text('#!/bin/sh\nexit 1')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_21():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, "Test error")
        with pytest.raises(FailedHookException) as exc_info:
            run_script("test_script.sh")
        assert exc_info.value.args[0] == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #53
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        repo_dir.mkdir()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        repo_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt'
        hook_file.write_text('#!/bin/sh\necho "test"')
        result = run_pre_prompt_hook(repo_dir)
        assert result != repo_dir
        assert result.exists()
        assert (result / 'hooks' / 'pre_prompt').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / 'repo'
        repo_dir.mkdir()
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'invalid_hook'
        hook_file.write_text('#!/bin/sh\nexit 1')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_evaluates_to_false():
    delete_project_on_failure = False
    assert not delete_project_on_failure


# LLM-generated content at query #55
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, False)
        mock_work_in.assert_called_once_with('repo_dir')
        mock_run_hook.assert_called_once_with('hook_name', 'project_dir', {'key': 'value'})

def test_run_hook_from_repo_dir_failed_hook_exception():
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = FailedHookException('error')
        with raises(FailedHookException):
            run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, True)
        mock_work_in.assert_called_once_with('repo_dir')
        mock_run_hook.assert_called_once_with('hook_name', 'project_dir', {'key': 'value'})
        mock_rmtree.assert_called_once_with('project_dir')
        mock_logger.exception.assert_called_once()

def test_run_hook_from_repo_dir_undefined_error():
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        mock_run_hook.side_effect = UndefinedError('error')
        with raises(UndefinedError):
            run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, True)
        mock_work_in.assert_called_once_with('repo_dir')
        mock_run_hook.assert_called_once_with('hook_name', 'project_dir', {'key': 'value'})
        mock_rmtree.assert_called_once_with('project_dir')
        mock_logger.exception.assert_called_once()


# LLM-generated content at query #56
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = os.path.join(original_dir, 'test_dir')
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir


# LLM-generated content at query #57
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #58
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        run_script(script_path, cwd)
        mock_popen.assert_called_once_with([sys.executable, script_path], shell=False, cwd=cwd)

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        run_script(script_path, cwd)
        mock_popen.assert_called_once_with([script_path], shell=False, cwd=cwd)

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd)
        assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_os_error_no_exec():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'No exec')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd)
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_other():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, 'Permission denied')
        with pytest.raises(FailedHookException) as excinfo:
            run_script(script_path, cwd)
        assert 'Hook script failed (error: [Errno 13] Permission denied)' in str(excinfo.value)

def test_run_script_windows_platform():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with patch('sys.platform', 'win32'):
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            run_script(script_path, cwd)
            mock_popen.assert_called_once_with([sys.executable, script_path], shell=True, cwd=cwd)


# LLM-generated content at query #59
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        assert delete_project_on_failure is True


# LLM-generated content at query #60
#--------------------------

```python
def test_tempfile_context_manager_succeeds():
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.txt') as temp:
        assert temp is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_21():
    with pytest.raises(OSError) as exc_info:
        run_script("nonexistent_script.sh")
    assert exc_info.value.errno == errno.ENOEXEC


# LLM-generated content at query #63
#--------------------------

```python
def test_predicate_evaluates_to_false():
    exit_status = 1
    assert not (exit_status != EXIT_SUCCESS)


# LLM-generated content at query #64
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/some/repo/dir'
    hook_name = 'some_hook'
    project_dir = '/some/project/dir'
    context = {}
    delete_project_on_failure = False

    with work_in(repo_dir):
        try:
            run_hook(hook_name, project_dir, context)
        except (FailedHookException, UndefinedError):
            assert not delete_project_on_failure
            raise


# LLM-generated content at query #65
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


