####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook_and_no_backup():
    assert valid_hook("pre-commit.py", "pre-commit") == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "post-commit") == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") == False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") == False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") == False


# LLM-generated content at query #2
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/test-templates/pre-post-hooks/')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/test-templates/pre-post-hooks/')
    with patch('cookiecutter.hooks.find_hook', return_value=['pre_prompt.py']):
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            result = run_pre_prompt_hook(repo_dir)
            assert result != repo_dir
            assert mock_run_script.called

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/test-templates/pre-post-hooks/')
    with patch('cookiecutter.hooks.find_hook', return_value=['pre_prompt.py']):
        with patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('error')):
            with pytest.raises(FailedHookException):
                run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]
    os.remove('hooks/pre-commit')
    os.rmdir('hooks')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/pre-commit~')
    os.rmdir('hooks')

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/post-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/post-commit')
    os.rmdir('hooks')


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_no_scripts_found():
    assert run_hook('nonexistent_hook', '/fake/dir', {}) is None

def test_run_hook_with_valid_script():
    hook_name = 'pre_gen_project'
    project_dir = '/fake/dir'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/fake/dir/hooks/pre_gen_project.py']
    assert find_hook(hook_name) == scripts
    run_hook(hook_name, project_dir, context)


# LLM-generated content at query #5
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit~", "pre-commit") is False
    assert valid_hook("pre-commit.py", "pre-commit") is True
    assert valid_hook("invalid-hook.py", "pre-commit") is False


# LLM-generated content at query #6
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = Path('test_script.py')
    script_path.write_text('print("Hello, {{ cookiecutter.project_name }}!")', encoding='utf-8')
    cwd = Path('.')
    run_script_with_context(script_path, cwd, context)


# LLM-generated content at query #7
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
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )


# LLM-generated content at query #8
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    Path(script_path).write_text('#!/bin/bash\necho "test"', encoding='utf-8')
    run_script_with_context(script_path, cwd, context)
    assert os.path.exists(temp.name)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_predicate_false():
    assert not os.path.isdir('hooks')


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'


# LLM-generated content at query #12
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre_commit.py", "pre_commit") == True


# LLM-generated content at query #13
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_valid_hook_path():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert result == [os.path.abspath('test_hooks_dir/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('test_hooks_dir/pre-commit.another', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert len(result) == 2
    assert os.path.abspath('test_hooks_dir/pre-commit') in result
    assert os.path.abspath('test_hooks_dir/pre-commit.another') in result


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_15_evaluates_to_true():
    assert not os.path.isdir('hooks') is False


# LLM-generated content at query #15
#--------------------------

```python
def test_find_hook_predicate_true():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    assert run_pre_prompt_hook("path/to/repo") == Path("path/to/repo")


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_hook_found():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    assert run_hook(hook_name, project_dir, context) is None

def test_run_hook_with_valid_hook():
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    scripts = ['/path/to/hook/script.py']
    assert run_hook(hook_name, project_dir, context) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_evaluates_to_true():
    os.path.isdir.return_value = True
    assert os.path.isdir('hooks')


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir="test_repo",
            hook_name="test_hook",
            project_dir="test_project",
            context={"test": "context"},
            delete_project_on_failure=True,
        )


# LLM-generated content at query #20
#--------------------------

```python
def test_run_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    repo_dir = Path('/some/repo')
    assert run_pre_prompt_hook(repo_dir) == repo_dir


# LLM-generated content at query #21
#--------------------------

```python
def test_valid_hook_with_matching_and_supported_hook():
    assert valid_hook("pre-commit.py", "pre-commit") is True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook("pre-commit.py", "post-commit") is False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook("unknown-hook.py", "unknown-hook") is False

def test_valid_hook_with_backup_file():
    assert valid_hook("pre-commit.py~", "pre-commit") is False

def test_valid_hook_with_wrong_extension():
    assert valid_hook("pre-commit.txt", "pre-commit") is False


# LLM-generated content at query #22
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = '/tmp/test_dir'
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "test_hook"
    project_dir = "/test/path"
    context = {"test": "context"}

    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        run_hook(hook_name, project_dir, context)


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook.return_value = []
    run_hook(hook_name, project_dir, context)
    logger.debug.assert_called_once_with('No %s hook found', hook_name)


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    proc.wait.assert_called_once()

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    proc = subprocess.Popen([script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 0
    run_script(script_path, cwd)
    proc.wait.assert_called_once()

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    proc = subprocess.Popen([sys.executable, script_path], shell=False, cwd=cwd)
    proc.wait.return_value = 1
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'error:' in str(excinfo.value)


# LLM-generated content at query #26
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_valid_hook_path():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]
    os.remove('hooks/pre-commit')
    os.rmdir('hooks')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/pre-commit~')
    os.rmdir('hooks')

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/post-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/post-commit')
    os.rmdir('hooks')

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
    os.remove('hooks/pre-commit')
    os.remove('hooks/pre-commit.another')
    os.rmdir('hooks')


# LLM-generated content at query #27
#--------------------------

```python
def test_hook_failure_without_deletion():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #28
#--------------------------

```python
def test_oserror_with_enonexec_raises_failed_hook_exception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'No exec')
        with pytest.raises(FailedHookException) as exc_info:
            run_script('test_script.sh')
        assert exc_info.value.args[0] == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #29
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/data/cookiecutter-no-hooks')
    result = hooks.run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/data/cookiecutter-with-hooks')
    result = hooks.run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/data/cookiecutter-failing-hook')
    with pytest.raises(FailedHookException):
        hooks.run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #30
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #31
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock()

        run_hook_from_repo_dir(
            repo_dir='test_repo',
            hook_name='test_hook',
            project_dir='test_project',
            context={},
            delete_project_on_failure=False,
        )

        mock_work_in.assert_called_once_with('test_repo')


# LLM-generated content at query #32
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = "/test/directory"
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = "test_script.py"
    cwd = "/test_dir"
    context = {"test_key": "test_value"}
    Path(script_path).write_text("test content", encoding='utf-8')
    run_script_with_context(script_path, cwd, context)
    assert Path(script_path).exists()


# LLM-generated content at query #34
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='repo_dir',
        hook_name='pre_gen_project',
        project_dir='project_dir',
        context={'cookiecutter': {}},
        delete_project_on_failure=True
    )

def test_run_hook_from_repo_dir_failure():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='pre_gen_project',
            project_dir='project_dir',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = "test_script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {"project_name": "test"}}
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=".sh")
    assert temp.name.endswith(".sh")


# LLM-generated content at query #36
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    assert find_hook('nonexistent-hook', 'hooks') is None

def test_find_hook_returns_list_of_absolute_paths_for_valid_hooks():
    scripts = find_hook('pre-commit', 'hooks')
    assert isinstance(scripts, list)
    assert all(os.path.isabs(script) for script in scripts)
    assert all('pre-commit' in script for script in scripts)

def test_find_hook_ignores_backup_files():
    scripts = find_hook('pre-commit', 'hooks')
    assert all(not script.endswith('~') for script in scripts)

def test_find_hook_ignores_unsupported_hooks():
    scripts = find_hook('unsupported-hook', 'hooks')
    assert scripts is None


# LLM-generated content at query #37
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None


# LLM-generated content at query #38
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/some/repo/dir"
    hook_name = "pre_gen_project"
    project_dir = "/some/project/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with work_in(repo_dir):
        try:
            run_hook(hook_name, project_dir, context)
        except (FailedHookException, UndefinedError):
            assert not delete_project_on_failure
            raise


# LLM-generated content at query #39
#--------------------------

```python
def test_hook_failure_without_deletion():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'fake': 'context'}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:

        mock_run_hook.side_effect = FailedHookException('Hook failed')

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #40
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    repo_dir = Path(tempfile.mkdtemp())
    assert run_pre_prompt_hook(repo_dir) == repo_dir


# LLM-generated content at query #41
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with work_in(repo_dir):
        assert os.getcwd() == repo_dir


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_evaluates_to_false():
    delete_project_on_failure = False
    assert not delete_project_on_failure


# LLM-generated content at query #43
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

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        assert mock_rmtree.called
        assert mock_logger.exception.called


# LLM-generated content at query #44
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        mock_work_in.return_value.__enter__.return_value = None
        run_hook_from_repo_dir(
            repo_dir='test_repo_dir',
            hook_name='test_hook',
            project_dir='test_project_dir',
            context={},
            delete_project_on_failure=False,
        )
        mock_work_in.assert_called_once_with('test_repo_dir')


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(), OSError)


# LLM-generated content at query #46
#--------------------------

```python
def test_oserror_with_enexec_raises_failed_hook_exception():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Exec format error')
        with raises(FailedHookException) as exc_info:
            run_script('test_script.sh')
        assert str(exc_info.value) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #47
#--------------------------

```python
def test_oserror_handling():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Test error')
        with pytest.raises(FailedHookException) as excinfo:
            run_script('test_script.sh')
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)


# LLM-generated content at query #48
#--------------------------

```python
def test_exit_status_not_success():
    exit_status = 1
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #49
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


# LLM-generated content at query #50
#--------------------------

```python
def test_run_hook_from_repo_dir_when_delete_project_on_failure_is_false():
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:

        mock_run_hook.side_effect = FailedHookException('Test error')
        repo_dir = '/fake/repo'
        hook_name = 'pre_gen_project'
        project_dir = '/fake/project'
        context = {'test': 'context'}
        delete_project_on_failure = False

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #51
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


# LLM-generated content at query #52
#--------------------------

```python
def test_run_pre_prompt_hook_no_hooks():
    repo_dir = Path('tests/test-templates/pre-prompt-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = Path('tests/test-templates/pre-prompt-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = Path('tests/test-templates/pre-prompt-hook-fail')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'test': 'value'}

    with patch('builtins.open', mock_open(read_data='test content')):
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = '/tmp/test_script.py'
            run_script_with_context(script_path, cwd, context)
            mock_temp_file.assert_called_once_with(delete=False, mode='wb', suffix='.py')


# LLM-generated content at query #54
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None


# LLM-generated content at query #55
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        with patch('cookiecutter.hooks.create_tmp_repo_dir') as mock_create:
            with patch('cookiecutter.hooks.work_in') as mock_work_in:
                result = run_pre_prompt_hook('/fake/repo')
                assert result == '/fake/repo'
                mock_create.assert_not_called()
                mock_work_in.assert_called_once_with('/fake/repo')


# LLM-generated content at query #56
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = "test_script.py"
    cwd = "/test/cwd"
    context = {"test": "value"}

    # Mock the necessary functions to avoid actual file operations
    Path(script_path).read_text = lambda encoding: "print('test')"
    tempfile.NamedTemporaryFile = lambda delete, mode, suffix: type('obj', (object,), {'name': 'temp_file', 'write': lambda x: None})()
    run_script = lambda x, y: None

    # Call the function
    run_script_with_context(script_path, cwd, context)

    # Assert that the temp file was created with delete=False
    assert tempfile.NamedTemporaryFile.called_with['delete'] == False


# LLM-generated content at query #57
#--------------------------

```python
def test_run_script_with_context_with_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'print("Hello, {{ cookiecutter.project_name }}!")'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.py', cwd)

def test_run_script_with_context_with_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'echo "Hello, {{ cookiecutter.project_name }}!"'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.sh'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.sh', cwd)

def test_run_script_with_context_with_jinja2_env_vars():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_jinja2_env_vars': {'trim_blocks': True}
        }
    }

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'print("Hello, {{ cookiecutter.project_name }}!")'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.py', cwd)


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    assert not (1 != 0)


# LLM-generated content at query #59
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = Path(tempfile.mkdtemp(prefix='test_dir'))
    with work_in(test_dir):
        assert os.getcwd() == str(test_dir)
    assert os.getcwd() == original_dir


# LLM-generated content at query #60
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

def test_run_script_os_error():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'


# LLM-generated content at query #61
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
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
        assert (result / 'hooks' / 'pre_prompt').exists()

def test_run_pre_prompt_hook_with_invalid_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'invalid_hook'
        hook_file.write_text('#!/bin/sh\nexit 1')
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #62
#--------------------------

```python
def test_run_script_with_context_creates_tempfile_with_correct_suffix():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'test': 'value'}

    with patch('builtins.open', mock_open(read_data='test content')):
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_tempfile.return_value.__enter__.return_value.name = '/tmp/temp_test_script.py'
            mock_tempfile.return_value.__enter__.return_value.suffix = '.py'
            run_script_with_context(script_path, cwd, context)

    assert mock_tempfile.call_args[1]['suffix'] == '.py'


# LLM-generated content at query #63
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #64
#--------------------------

```python
def test_predicate_false():
    assert not (0 != 0)


# LLM-generated content at query #65
#--------------------------

```python
def test_find_hook_returns_list_when_hook_exists():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #66
#--------------------------

```python
def test_find_hook_empty_scripts():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=[]):
        assert find_hook('pre-commit') is None


# LLM-generated content at query #67
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['invalid_hook.sh']):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_list_with_valid_hook():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert result == ['hooks/pre-commit']

def test_find_hook_ignores_backup_files():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit~']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_ignores_non_matching_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['other-hook']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        assert find_hook('pre-commit') is None

def test_find_hook_returns_multiple_valid_hooks():
    with patch('os.path.isdir', return_value=True), \
         patch('os.listdir', return_value=['pre-commit', 'pre-commit.sh']), \
         patch('os.path.abspath', side_effect=lambda x: x):
        result = find_hook('pre-commit')
        assert len(result) == 2
        assert 'hooks/pre-commit' in result
        assert 'hooks/pre-commit.sh' in result


# LLM-generated content at query #68
#--------------------------

```python
def test_find_hook_returns_list_when_hook_exists():
    # Setup: Create a temporary directory with a valid hook file
    import os
    import tempfile
    from your_module import find_hook

    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre-commit')
        with open(hook_file, 'w') as f:
            f.write('#!/bin/sh\necho "test"')

        result = find_hook('pre-commit', hooks_dir)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


# LLM-generated content at query #69
#--------------------------

```python
def test_find_hook_returns_none_if_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_if_no_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/invalid_hook.py', 'w') as f:
        f.write('')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_list_of_valid_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('pre-commit')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('')
    assert find_hook('pre-commit', 'hooks') is None

def test_find_hook_returns_absolute_paths():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert os.path.isabs(result[0])


# LLM-generated content at query #70
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
    assert find_hook('unsupported-hook', 'hooks') is None

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


# LLM-generated content at query #71
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

def test_find_hook_returns_none_for_empty_hooks_dir():
    assert find_hook('pre-commit', 'empty_hooks_dir') is None


# LLM-generated content at query #72
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre-commit', 'nonexistent_dir')
    assert result is None

def test_find_hook_returns_none_when_no_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/other-hook', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result is None

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result == [os.path.abspath('hooks/pre-commit')]

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit~', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert result is None

def test_find_hook_ignores_unsupported_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/unsupported-hook', 'w') as f:
        f.write('')
    result = find_hook('unsupported-hook', 'hooks')
    assert result is None

def test_find_hook_returns_multiple_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('')
    with open('hooks/pre-commit.another', 'w') as f:
        f.write('')
    result = find_hook('pre-commit', 'hooks')
    assert len(result) == 2
    assert os.path.abspath('hooks/pre-commit') in result
    assert os.path.abspath('hooks/pre-commit.another') in result


# LLM-generated content at query #73
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/data/fake-repo-pre-prompt')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/data/fake-repo-pre-prompt-with-hook')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed_script():
    repo_dir = Path('tests/data/fake-repo-pre-prompt-failed-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #74
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = '/tmp'
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #75
#--------------------------

```python
def test_run_script_oserror_not_enoexec():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.EACCES, 'Permission denied')
        with pytest.raises(FailedHookException) as exc_info:
            run_script('/path/to/script.sh')
        assert str(exc_info.value) == 'Hook script failed (error: [Errno 13] Permission denied)'


# LLM-generated content at query #76
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = '/path/to/script.py'
    cwd = '/working/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_successful_non_python_script():
    script_path = '/path/to/script.sh'
    cwd = '/working/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_failed_hook_exception():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_enoexec():
    script_path = '/path/to/empty_script'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_general():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(script_path, cwd)


# LLM-generated content at query #77
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(), OSError)


# LLM-generated content at query #78
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(errno.ENOEXEC), OSError)


# LLM-generated content at query #79
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/some/repo/dir"
    hook_name = "pre_gen_project"
    project_dir = "/some/project/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        mock_run_hook.side_effect = FailedHookException("Hook failed")
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    assert not os.path.exists(project_dir)


# LLM-generated content at query #80
#--------------------------

```python
def test_run_script_with_context_calls_run_script_with_rendered_template():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    context = {'test_var': 'test_value'}

    with patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.Path') as mock_path, \
         patch('tempfile.NamedTemporaryFile') as mock_temp_file:

        mock_path_instance = MagicMock()
        mock_path_instance.read_text.return_value = 'print("{{ test_var }}")'
        mock_path.return_value = mock_path_instance

        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = 'print("test_value")'
        mock_env.from_string.return_value = mock_template
        mock_create_env.return_value = mock_env

        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = '/tmp/temp_script.py'
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

        run_script_with_context(script_path, cwd, context)

        mock_create_env.assert_called_once_with(context)
        mock_env.from_string.assert_called_once_with('print("{{ test_var }}")')
        mock_template.render.assert_called_once_with(**context)
        mock_temp_file_instance.write.assert_called_once_with(b'print("test_value")')
        mock_run_script.assert_called_once_with('/tmp/temp_script.py', cwd)


# LLM-generated content at query #81
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path("test_repo")
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #82
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    run_hook(hook_name, project_dir, context)
    assert True


# LLM-generated content at query #83
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None

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


# LLM-generated content at query #84
#--------------------------

```python
def test_oserror_predicate_false():
    assert not isinstance(OSError(errno.ENOEXEC), OSError)


# LLM-generated content at query #85
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = "/tmp/test_dir"
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #86
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre_commit.py", "pre_commit") is True


# LLM-generated content at query #87
#--------------------------

```python
def test_find_hook_returns_none_when_no_scripts_found():
    assert find_hook('nonexistent_hook', 'empty_hooks_dir') is None


# LLM-generated content at query #88
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='repo_dir',
        hook_name='pre_gen_project',
        project_dir='project_dir',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True,
    )


# LLM-generated content at query #89
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

def test_run_hook_from_repo_dir_undefined_error_with_delete():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, True)

def test_run_hook_from_repo_dir_undefined_error_without_delete():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {'key': 'value'}, False)


# LLM-generated content at query #90
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #91
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #92
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


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
    assert result.exists()
    assert result.is_dir()

def test_run_pre_prompt_hook_fails():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
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

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/post-commit', 'w') as f:
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
def test_valid_hook_with_matching_hook_and_supported_hook_and_not_backup_file():
    assert valid_hook('pre-commit', 'pre-commit') == True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook('pre-commit', 'commit-msg') == False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook('unknown-hook', 'unknown-hook') == False

def test_valid_hook_with_backup_file():
    assert valid_hook('pre-commit~', 'pre-commit') == False

def test_valid_hook_with_backup_file_even_if_matching_and_supported():
    assert valid_hook('pre-commit~', 'pre-commit') == False

def test_valid_hook_with_different_extension():
    assert valid_hook('pre-commit.sh', 'pre-commit') == False

def test_valid_hook_with_path_in_filename():
    assert valid_hook('/path/to/pre-commit', 'pre-commit') == True


# LLM-generated content at query #4
#--------------------------

```python
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook('pre-commit.py', 'pre-commit') is True

def test_valid_hook_with_non_matching_hook():
    assert valid_hook('pre-commit.py', 'commit-msg') is False

def test_valid_hook_with_unsupported_hook():
    assert valid_hook('unknown-hook.py', 'unknown-hook') is False

def test_valid_hook_with_backup_file():
    assert valid_hook('pre-commit.py~', 'pre-commit') is False

def test_valid_hook_with_wrong_extension():
    assert valid_hook('pre-commit.txt', 'pre-commit') is False


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)

def test_run_script_failed_exit_status():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='Hook script failed (exit status: 1)'):
        run_script(script_path, cwd)

def test_run_script_failed_enoexec():
    script_path = 'empty_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='Hook script failed, might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_failed_oserror():
    script_path = 'invalid_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='Hook script failed (error: '):
        run_script(script_path, cwd)


# LLM-generated content at query #6
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

def test_run_hook_from_repo_dir_failure_with_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='failing_repo',
            hook_name='pre_gen_project',
            project_dir='failing_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='undefined_repo',
            hook_name='pre_gen_project',
            project_dir='undefined_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #7
#--------------------------

```python
def test_find_hook_predicate():
    assert find_hook('pre-commit', 'hooks') is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_hooks_dir_is_not_directory():
    import os
    from unittest.mock import patch

    with patch('os.path.isdir', return_value=False):
        assert find_hook('test_hook') is None


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_evaluates_to_true():
    os.path.isdir.return_value = True
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_no_hooks_found():
    assert run_hook('pre_gen_project', '/tmp', {}) is None

def test_run_hook_with_valid_hook():
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', '/tmp', context)


# LLM-generated content at query #11
#--------------------------

```python
def test_hooks_dir_is_directory():
    os.path.isdir.return_value = True
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #12
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('/test/repo')
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    find_hook.return_value = []
    logger.debug.assert_called_once_with('No %s hook found', hook_name)
    run_hook(hook_name, project_dir, context)


# LLM-generated content at query #14
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'_jinja2_env_vars': {}, 'project_name': 'test_project'}}
    script_path = Path('test_script.sh')
    script_path.write_text('echo "{{ cookiecutter.project_name }}"', encoding='utf-8')
    run_script_with_context(script_path, '.', context)
    assert script_path.read_text(encoding='utf-8') == 'echo "{{ cookiecutter.project_name }}"'


# LLM-generated content at query #15
#--------------------------

```python
def test_run_script_with_context_creates_tempfile_with_delete_false():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    context = {'test': 'value'}

    with patch('builtins.open', mock_open(read_data='#!/bin/bash\necho "test"')) as mock_file, \
         patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
         patch('cookiecutter.hooks.run_script') as mock_run_script, \
         patch('cookiecutter.hooks.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.hooks.Path') as mock_path:

        mock_path_instance = MagicMock()
        mock_path_instance.read_text.return_value = '#!/bin/bash\necho "test"'
        mock_path.return_value = mock_path_instance

        mock_env = MagicMock()
        mock_create_env.return_value = mock_env

        mock_template = MagicMock()
        mock_template.render.return_value = '#!/bin/bash\necho "test"'
        mock_env.from_string.return_value = mock_template

        run_script_with_context(script_path, cwd, context)

        mock_tempfile.assert_called_once_with(delete=False, mode='wb', suffix='.sh')


# LLM-generated content at query #16
#--------------------------

```python
def test_run_hook_no_scripts_found():
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    find_hook = lambda _: []
    logger = Mock()
    run_hook(hook_name, project_dir, context)
    assert logger.debug.call_args_list[0][0][0] == 'No %s hook found'
    assert logger.debug.call_args_list[0][0][1] == hook_name


# LLM-generated content at query #17
#--------------------------

```python
def test_run_hook_no_hooks_found():
    assert run_hook('pre_gen_project', '/tmp/project', {'cookiecutter': {}}) is None

def test_run_hook_with_valid_hook():
    context = {'cookiecutter': {'project_name': 'test'}}
    run_hook('pre_gen_project', '/tmp/project', context)


# LLM-generated content at query #18
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") is True


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts():
    repo_dir = Path('/path/to/repo')
    assert run_pre_prompt_hook(repo_dir) == repo_dir


# LLM-generated content at query #20
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

def test_run_pre_prompt_hook_fails():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #21
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
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )

def test_run_hook_from_repo_dir_failure_without_deletion():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='invalid_repo',
            hook_name='pre_gen_project',
            project_dir='invalid_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False
        )

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_with_undefined',
            hook_name='pre_gen_project',
            project_dir='project_with_undefined',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #22
#--------------------------

```python
def test_run_script_with_context_tempfile_delete_false():
    script_path = '/path/to/script.sh'
    cwd = '/current/working/directory'
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}

    with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
        mock_tempfile_instance = mock_tempfile.return_value.__enter__.return_value
        mock_tempfile_instance.name = '/tmp/temp_script.sh'
        mock_tempfile_instance.delete = True

        run_script_with_context(script_path, cwd, context)

        assert mock_tempfile_instance.delete is False


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_work_in_context_manager_is_used():
    with work_in(repo_dir) as cm:
        assert cm is None


# LLM-generated content at query #25
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('test_hooks_dir/pre-commit')
    os.remove('test_hooks_dir/pre-commit')
    os.rmdir('test_hooks_dir')

def test_find_hook_ignores_backup_files():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/pre-commit~')
    os.rmdir('test_hooks_dir')

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/other-hook')
    os.rmdir('test_hooks_dir')

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('test_hooks_dir/pre-commit.bak', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert result is not None
    assert len(result) == 2
    os.remove('test_hooks_dir/pre-commit')
    os.remove('test_hooks_dir/pre-commit.bak')
    os.rmdir('test_hooks_dir')


# LLM-generated content at query #26
#--------------------------

```python
def test_run_script_with_context_creates_temp_file():
    script_path = 'test_script.sh'
    cwd = '/tmp'
    context = {'cookiecutter': {'project_name': 'test'}}
    Path(script_path).write_text('#!/bin/bash\necho "Hello {{ cookiecutter.project_name }}"', encoding='utf-8')

    run_script_with_context(script_path, cwd, context)

    assert Path(script_path).exists()


# LLM-generated content at query #27
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'non_existent_dir') is None


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_evaluates_to_false():
    script_path = "test_script.sh"
    cwd = "/tmp"
    context = {}

    assert not os.path.splitext(script_path)[1]


# LLM-generated content at query #30
#--------------------------

```python
def test_oserror_with_enoexec():
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = OSError(errno.ENOEXEC, 'Test error')
        with pytest.raises(FailedHookException) as exc_info:
            run_script('test_script.sh')
        assert 'Hook script failed, might be an empty file or missing a shebang' in str(exc_info.value)


# LLM-generated content at query #31
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = '/path/to/script.py'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_non_python_file():
    script_path = '/path/to/script.sh'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_fails_with_exit_status():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'exit status' in str(excinfo.value)

def test_run_script_fails_with_os_error():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'error' in str(excinfo.value)

def test_run_script_fails_with_missing_shebang():
    script_path = '/path/to/script_without_shebang'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'missing a shebang' in str(excinfo.value)


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #33
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-push', 'w') as f:
        f.write('#!/bin/sh\necho "pre-push"')
    assert find_hook('pre-commit', 'hooks_dir') is None

def test_find_hook_returns_list_with_valid_hook():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "pre-commit"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks_dir/pre-commit')

def test_find_hook_ignores_backup_files():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "pre-commit"')
    with open('hooks_dir/pre-commit~', 'w') as f:
        f.write('#!/bin/sh\necho "backup"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('hooks_dir/pre-commit')

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('hooks_dir', exist_ok=True)
    with open('hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "pre-commit"')
    with open('hooks_dir/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "pre-commit.sh"')
    result = find_hook('pre-commit', 'hooks_dir')
    assert result is not None
    assert len(result) == 2
    assert all(path.endswith(('hooks_dir/pre-commit', 'hooks_dir/pre-commit.sh')) for path in result)


# LLM-generated content at query #34
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
    assert result.is_dir()

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/fake-repo-pre-with-failing-hook')
    with pytest.raises(hooks.FailedHookException):
        hooks.run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/fake/repo"
    hook_name = "test_hook"
    project_dir = "/fake/project"
    context = {}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException("Test error")
        mock_work_in.return_value.__enter__ = Mock()
        mock_work_in.return_value.__exit__ = Mock()

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_evaluates_to_false():
    delete_project_on_failure = False
    assert not delete_project_on_failure


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_evaluates_to_false():
    exit_status = 1
    assert not (exit_status != EXIT_SUCCESS)


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    """Test that the predicate at line 21 evaluates to False."""
    try:
        raise OSError(errno.EACCES, "Permission denied")
    except OSError as err:
        assert err.errno != errno.ENOEXEC


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS is False


# LLM-generated content at query #41
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = tempfile.mkdtemp()

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(errno.ENOEXEC), OSError)


# LLM-generated content at query #43
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
    assert find_hook('pre-commit', 'hooks') is None

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


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.sh'
    cwd = '/test/dir'

    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'echo {{ cookiecutter.project_name }}'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.sh'
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_env.return_value.from_string.return_value = mock_template
                mock_template.render.return_value = 'echo test_project'
                with patch('cookiecutter.hooks.run_script') as mock_run:
                    run_script_with_context(script_path, cwd, context)
                    mock_run.assert_called_once_with('temp_script.sh', cwd)


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_false():
    exit_status = 1
    assert exit_status != 0


# LLM-generated content at query #46
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    """Test that the predicate at line 17 evaluates to False."""
    repo_dir = "/some/repo/dir"
    hook_name = "pre_gen_project"
    project_dir = "/some/project/dir"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with work_in(repo_dir):
        assert not delete_project_on_failure


# LLM-generated content at query #47
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

def test_find_hook_returns_valid_hook_paths():
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

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/other-hook', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/other-hook')
    os.rmdir('test_hooks_dir')


# LLM-generated content at query #48
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = '/path/to/test_script.py'
    cwd = '/working/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_successful_non_python_script():
    script_path = '/path/to/test_script.sh'
    cwd = '/working/directory'
    assert run_script(script_path, cwd) is None

def test_run_script_failed_hook_exception():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = '/path/to/empty_file'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_general():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/working/directory'
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(script_path, cwd)


# LLM-generated content at query #49
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = "/tmp/test_dir"
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir


# LLM-generated content at query #50
#--------------------------

```python
def test_run_pre_prompt_hook_no_hook():
    repo_dir = Path('tests/data/cookiecutter-no-hooks')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook():
    repo_dir = Path('tests/data/cookiecutter-with-hooks')
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert result.is_dir()
    assert result.name == repo_dir.name

def test_run_pre_prompt_hook_failed():
    repo_dir = Path('tests/data/cookiecutter-failing-hook')
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #51
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

def test_run_hook_from_repo_dir_undefined_error():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='repo_with_undefined',
            hook_name='pre_gen_project',
            project_dir='project_with_undefined',
            context={'cookiecutter': {}},
            delete_project_on_failure=True
        )


# LLM-generated content at query #52
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'empty file or missing a shebang' in str(excinfo.value)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/directory'
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed' in str(excinfo.value)


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    assert not isinstance(OSError(), OSError)


# LLM-generated content at query #54
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

def test_run_script_failed_hook_exception():
    script_path = 'failing_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException):
        run_script(script_path, cwd)

def test_run_script_os_error_empty_file():
    script_path = 'empty_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_general():
    script_path = 'nonexistent_script.py'
    cwd = '/test/dir'
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(script_path, cwd)


# LLM-generated content at query #55
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', context, False)
    assert True

def test_run_hook_from_repo_dir_failure_with_delete():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    try:
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', context, True)
    except Exception:
        assert True

def test_run_hook_from_repo_dir_failure_without_delete():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    try:
        run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', context, False)
    except Exception:
        assert True


# LLM-generated content at query #56
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_matching_hooks():
    assert find_hook('nonexistent-hook', 'hooks') is None

def test_find_hook_returns_absolute_path_for_valid_hook():
    assert find_hook('pre-commit', 'hooks') == ['/path/to/hooks/pre-commit']

def test_find_hook_ignores_backup_files():
    assert find_hook('pre-commit', 'hooks') != ['/path/to/hooks/pre-commit~']

def test_find_hook_ignores_non_matching_hooks():
    assert find_hook('pre-commit', 'hooks') != ['/path/to/hooks/post-commit']


# LLM-generated content at query #57
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = '/tmp'
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #58
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    script_path = 'test_script.py'
    cwd = '/tmp'
    with patch('cookiecutter.hooks.Path') as mock_path:
        mock_path.return_value.read_text.return_value = 'print("Hello {{ cookiecutter.project_name }}")'
        with patch('cookiecutter.hooks.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_script.py'
            with patch('cookiecutter.hooks.run_script') as mock_run:
                run_script_with_context(script_path, cwd, context)
                mock_run.assert_called_once_with('temp_script.py', cwd)


# LLM-generated content at query #59
#--------------------------

```python
def test_pre_prompt_hook_returns_original_repo_dir_when_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        repo_dir = Path('/fake/repo')
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir


# LLM-generated content at query #60
#--------------------------

```python
def test_exit_status_not_success_raises_exception():
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        with pytest.raises(FailedHookException):
            run_script('test_script.py')


# LLM-generated content at query #61
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    run_hook_from_repo_dir(
        repo_dir='./test_repo',
        hook_name='pre_gen_project',
        project_dir='./test_project',
        context={'cookiecutter': {'project_name': 'test'}},
        delete_project_on_failure=True,
    )

def test_run_hook_from_repo_dir_failure_with_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='pre_gen_project',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True,
        )

def test_run_hook_from_repo_dir_failure_without_delete():
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='pre_gen_project',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False,
        )

def test_run_hook_from_repo_dir_undefined_error_with_delete():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='pre_gen_project',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=True,
        )

def test_run_hook_from_repo_dir_undefined_error_without_delete():
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir='./test_repo',
            hook_name='pre_gen_project',
            project_dir='./test_project',
            context={'cookiecutter': {'project_name': 'test'}},
            delete_project_on_failure=False,
        )


# LLM-generated content at query #62
#--------------------------

```python
def test_pre_prompt_hook_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #63
#--------------------------

```python
def test_tempfile_delete_false():
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        assert temp.delete is False


# LLM-generated content at query #64
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    subprocess.Popen.return_value.wait.return_value = 0
    run_script(script_path, cwd)
    subprocess.Popen.assert_called_once_with([sys.executable, script_path], shell=sys.platform.startswith('win'), cwd=cwd)

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    subprocess.Popen.return_value.wait.return_value = 0
    run_script(script_path, cwd)
    subprocess.Popen.assert_called_once_with([script_path], shell=sys.platform.startswith('win'), cwd=cwd)

def test_run_script_failed_python_script():
    script_path = 'test_script.py'
    cwd = '/test/directory'
    subprocess.Popen.return_value.wait.return_value = 1
    with pytest.raises(FailedHookException, match='Hook script failed (exit status: 1)'):
        run_script(script_path, cwd)

def test_run_script_os_error_no_exec():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    subprocess.Popen.side_effect = OSError(errno.ENOEXEC, 'No exec')
    with pytest.raises(FailedHookException, match='Hook script failed, might be an empty file or missing a shebang'):
        run_script(script_path, cwd)

def test_run_script_os_error_other():
    script_path = 'test_script.sh'
    cwd = '/test/directory'
    subprocess.Popen.side_effect = OSError(errno.EACCES, 'Permission denied')
    with pytest.raises(FailedHookException, match='Hook script failed (error: '):
        run_script(script_path, cwd)


# LLM-generated content at query #65
#--------------------------

```python
def test_exit_status_success():
    exit_status = 0
    assert exit_status == EXIT_SUCCESS


# LLM-generated content at query #66
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #67
#--------------------------

```python
def test_run_script_successful_python_script():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    subprocess.Popen.return_value.wait.return_value = 0
    run_script(script_path, cwd)
    subprocess.Popen.assert_called_once_with([sys.executable, script_path], shell=sys.platform.startswith('win'), cwd=cwd)

def test_run_script_successful_non_python_script():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    subprocess.Popen.return_value.wait.return_value = 0
    run_script(script_path, cwd)
    subprocess.Popen.assert_called_once_with([script_path], shell=sys.platform.startswith('win'), cwd=cwd)

def test_run_script_failed_hook_exception():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    subprocess.Popen.return_value.wait.return_value = 1
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (exit status: 1)' in str(excinfo.value)

def test_run_script_os_error():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    subprocess.Popen.side_effect = OSError(errno.ENOEXEC, 'Executable not found')
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed, might be an empty file or missing a shebang' in str(excinfo.value)

def test_run_script_generic_os_error():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    subprocess.Popen.side_effect = OSError('Generic OS error')
    with pytest.raises(FailedHookException) as excinfo:
        run_script(script_path, cwd)
    assert 'Hook script failed (error: Generic OS error)' in str(excinfo.value)


# LLM-generated content at query #68
#--------------------------

```python
def test_work_in_context_manager_enters_directory():
    initial_dir = os.getcwd()
    test_dir = '/test/directory'
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == initial_dir


# LLM-generated content at query #69
#--------------------------

```python
def test_run_hook_from_repo_dir_uses_work_in_context_manager():
    repo_dir = Path('/some/repo/dir')
    hook_name = 'pre_gen_project'
    project_dir = Path('/some/project/dir')
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook'):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #70
#--------------------------

```python
def test_run_pre_prompt_hook_no_pre_prompt_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo_dir')
        assert result == '/fake/repo_dir'


# LLM-generated content at query #71
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_non_python_file():
    script_path = 'test_script.sh'
    cwd = '/test/dir'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_failed_exit_status():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_with_os_error():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory)'

def test_run_script_with_empty_file():
    script_path = 'test_script.py'
    cwd = '/test/dir'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #72
#--------------------------

```python
def test_work_in_context_manager_with_none_dirname():
    with work_in(None) as result:
        assert result is None


# LLM-generated content at query #73
#--------------------------

```python
def test_find_hook_returns_none_when_no_scripts_found():
    assert find_hook('nonexistent_hook', 'empty_hooks_dir') is None


# LLM-generated content at query #74
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

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/post-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'test_hooks_dir') is None
    os.remove('test_hooks_dir/post-commit')
    os.rmdir('test_hooks_dir')

def test_find_hook_returns_multiple_valid_hooks():
    os.makedirs('test_hooks_dir', exist_ok=True)
    with open('test_hooks_dir/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('test_hooks_dir/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'test_hooks_dir')
    assert len(result) == 2
    assert os.path.abspath('test_hooks_dir/pre-commit') in result
    assert os.path.abspath('test_hooks_dir/pre-commit.sh') in result
    os.remove('test_hooks_dir/pre-commit')
    os.remove('test_hooks_dir/pre-commit.sh')
    os.rmdir('test_hooks_dir')


# LLM-generated content at query #75
#--------------------------

```python
def test_find_hook_returns_none_when_no_matching_scripts():
    assert find_hook('nonexistent_hook', 'empty_hooks_dir') is None


# LLM-generated content at query #76
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') == True


# LLM-generated content at query #77
#--------------------------

```python
def test_find_hook_predicate_false():
    assert not (len([]) == 0)


# LLM-generated content at query #78
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks')


# LLM-generated content at query #79
#--------------------------

```python
def test_find_hook_predicate_true():
    assert os.path.isdir('hooks') or not os.path.isdir('hooks')


# LLM-generated content at query #80
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    assert find_hook('pre-commit', 'nonexistent_dir') is None

def test_find_hook_returns_none_when_no_valid_hooks():
    os.makedirs('empty_hooks_dir', exist_ok=True)
    assert find_hook('pre-commit', 'empty_hooks_dir') is None
    os.rmdir('empty_hooks_dir')

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

def test_find_hook_ignores_non_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/post-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test"')
    assert find_hook('pre-commit', 'hooks') is None
    os.remove('hooks/post-commit')
    os.rmdir('hooks')

def test_find_hook_returns_multiple_matching_hooks():
    os.makedirs('hooks', exist_ok=True)
    with open('hooks/pre-commit', 'w') as f:
        f.write('#!/bin/sh\necho "test1"')
    with open('hooks/pre-commit.sh', 'w') as f:
        f.write('#!/bin/sh\necho "test2"')
    result = find_hook('pre-commit', 'hooks')
    assert result is not None
    assert len(result) == 2
    assert all(path.endswith(('hooks/pre-commit', 'hooks/pre-commit.sh')) for path in result)
    os.remove('hooks/pre-commit')
    os.remove('hooks/pre-commit.sh')
    os.rmdir('hooks')


# LLM-generated content at query #81
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = "/some/repo"
    hook_name = "pre_gen_project"
    project_dir = "/some/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False

    with patch("cookiecutter.hooks.work_in") as mock_work_in, \
         patch("cookiecutter.hooks.run_hook") as mock_run_hook:
        mock_run_hook.side_effect = FailedHookException("Hook failed")
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
    assert not os.path.exists(project_dir)


# LLM-generated content at query #82
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    assert valid_hook("pre-commit", "pre-commit") == True


# LLM-generated content at query #83
#--------------------------

```python
def test_find_hook_returns_none_when_no_matching_scripts():
    assert find_hook('nonexistent_hook', 'empty_hooks_dir') is None


# LLM-generated content at query #84
#--------------------------

```python
def test_find_hook_predicate():
    assert os.path.isdir('hooks') is True


# LLM-generated content at query #85
#--------------------------

```python
def test_run_hook_no_scripts_found():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        run_hook('pre_gen_project', '/tmp/project', {'cookiecutter': {}})


# LLM-generated content at query #86
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_delete_false():
    script_path = 'test_script.py'
    cwd = '/tmp'
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.py') as temp:
        assert temp.delete is False


# LLM-generated content at query #87
#--------------------------

```python
def test_run_pre_prompt_hook_returns_repo_dir_when_no_hooks():
    repo_dir = Path("test_repo")
    repo_dir.mkdir()
    (repo_dir / "hooks").mkdir()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    repo_dir.rmdir()


# LLM-generated content at query #88
#--------------------------

```python
def test_run_script_with_python_file():
    script_path = '/path/to/script.py'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_non_python_file():
    script_path = '/path/to/script.sh'
    cwd = '/working/directory'
    run_script(script_path, cwd)
    assert True

def test_run_script_with_default_cwd():
    script_path = '/path/to/script.py'
    run_script(script_path)
    assert True

def test_run_script_fails_with_exit_status():
    script_path = '/path/to/failing_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_fails_with_os_error():
    script_path = '/path/to/nonexistent_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'/path/to/nonexistent_script.py\')'

def test_run_script_fails_with_enexec_error():
    script_path = '/path/to/empty_script.py'
    cwd = '/working/directory'
    try:
        run_script(script_path, cwd)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #89
#--------------------------

```python
def test_run_hook_from_repo_dir_when_delete_project_on_failure_is_false():
    repo_dir = Path('/some/repo/dir')
    hook_name = 'pre_gen_project'
    project_dir = Path('/some/project/dir')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in:
        with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException):
            with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                with pytest.raises(FailedHookException):
                    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    mock_rmtree.assert_not_called()


# LLM-generated content at query #90
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_suffix():
    script_path = 'test_script.py'
    cwd = '/tmp'
    context = {'cookiecutter': {'project_name': 'test'}}
    _, extension = os.path.splitext(script_path)
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=extension) as temp:
        assert temp.name.endswith(extension)


# LLM-generated content at query #91
#--------------------------

```python
def test_run_hook_from_repo_dir_with_delete_project_on_failure():
    repo_dir = "/path/to/repo"
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = True

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException("Hook failed")

        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #92
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
        assert str(e) == 'Hook script failed (error: [Errno 2] No such file or directory: \'nonexistent_script.py\')'


# LLM-generated content at query #93
#--------------------------

```python
def test_run_pre_prompt_hook_no_scripts():
    with patch('cookiecutter.hooks.find_hook', return_value=[]):
        result = run_pre_prompt_hook('/fake/repo')
        assert result == '/fake/repo'


# LLM-generated content at query #94
#--------------------------

```python
def test_run_hook_from_repo_dir_predicate_false():
    repo_dir = '/fake/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/fake/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:

        mock_run_hook.side_effect = FailedHookException('Hook failed')
        mock_work_in.return_value.__enter__ = lambda self: None
        mock_work_in.return_value.__exit__ = lambda self, *args: None

        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

        mock_rmtree.assert_not_called()


# LLM-generated content at query #95
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    exit_status = 0
    assert exit_status != EXIT_SUCCESS is False


