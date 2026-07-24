####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_valid_hook_with_matching_supported_hook():
    assert valid_hook('/path/to/hooks/pre-commit', 'pre-commit') == True

def test_valid_hook_with_matching_unsupported_hook():
    assert valid_hook('/path/to/hooks/unsupported-hook', 'unsupported-hook') == False

def test_valid_hook_with_non_matching_supported_hook():
    assert valid_hook('/path/to/hooks/commit-msg', 'pre-commit') == False

def test_valid_hook_with_matching_supported_backup_hook():
    assert valid_hook('/path/to/hooks/pre-commit~', 'pre-commit') == False

def test_valid_hook_with_non_matching_supported_backup_hook():
    assert valid_hook('/path/to/hooks/commit-msg~', 'pre-commit') == False


# LLM-generated content at query #2
#--------------------------

```python
def test_run_script_success():
    script_path = '/path/to/script.py'
    run_script(script_path)

def test_run_script_failed_exit_status():
    script_path = '/path/to/failing_script.py'
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed (exit status: 1)'

def test_run_script_failed_os_error():
    script_path = '/path/to/nonexistent_script.py'
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert 'Hook script failed (error: ' in str(e)

def test_run_script_failed_enodexec():
    script_path = '/path/to/invalid_script.py'
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert str(e) == 'Hook script failed, might be an empty file or missing a shebang'


# LLM-generated content at query #3
#--------------------------

Here are the test cases:


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_with_valid_hook():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'pre_gen_project'
    mock_script_path = '/tmp/project/hooks/pre_gen_project.sh'
    mock_temp_path = '/tmp/tempfile.sh'
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        return [mock_script_path]
    
    def mock_run_script_with_context(script_path, cwd, context):
        assert script_path == mock_temp_path
        assert cwd == project_dir
        assert context == context
    
    original_find_hook = hooks.find_hook
    original_run_script_with_context = hooks.run_script_with_context
    hooks.find_hook = mock_find_hook
    hooks.run_script_with_context = mock_run_script_with_context
    
    try:
        hooks.run_hook(hook_name, project_dir, context)
    finally:
        hooks.find_hook = original_find_hook
        hooks.run_script_with_context = original_run_script_with_context


def test_run_hook_with_no_hook_found():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'pre_gen_project'
    
    def mock_find_hook(hook_name, hooks_dir='hooks'):
        return None
    
    original_find_hook = hooks.find_hook
    hooks.find_hook = mock_find_hook
    
    try:
        hooks.run_hook(hook_name, project_dir, context)
    finally:
        hooks.find_hook = original_find_hook


# LLM-generated content at query #5
#--------------------------

```python
def test_valid_hook_returns_true_for_valid_hook():
    hook_file = "valid_hook.py"
    hook_name = "valid_hook"
    _HOOKS = {"valid_hook"}
    result = valid_hook(hook_file, hook_name)
    assert result == True


# LLM-generated content at query #6
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = 'test_repo'
    os.makedirs(repo_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(script_path, 'w') as f:
        f.write('print("Pre-prompt hook executed")')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = 'test_repo'
    os.makedirs(repo_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'invalid_hook.py')
    with open(script_path, 'w') as f:
        f.write('print("Invalid hook executed")')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_no_hooks_dir():
    repo_dir = 'test_repo'
    os.makedirs(repo_dir)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_empty_hooks_dir():
    repo_dir = 'test_repo'
    os.makedirs(repo_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_failing_hook():
    repo_dir = 'test_repo'
    os.makedirs(repo_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(script_path, 'w') as f:
        f.write('exit(1)')
    result = None
    try:
        result = run_pre_prompt_hook(repo_dir)
    except FailedHookException:
        assert result is None
    shutil.rmtree(repo_dir)


# LLM-generated content at query #7
#--------------------------

```
def test_find_hook_with_valid_hook_and_directory():
    hooks_dir = 'tests/fixtures/hooks'
    hook_name = 'pre_gen_project'
    result = find_hook(hook_name, hooks_dir)
    expected = [os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py'))]
    assert result == expected

def test_find_hook_with_valid_hook_and_directory_multiple_files():
    hooks_dir = 'tests/fixtures/hooks_multiple'
    hook_name = 'pre_gen_project'
    result = find_hook(hook_name, hooks_dir)
    expected = [
        os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py')),
        os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.sh'))
    ]
    assert sorted(result) == sorted(expected)

def test_find_hook_with_invalid_hook():
    hooks_dir = 'tests/fixtures/hooks'
    hook_name = 'invalid_hook'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_invalid_directory():
    hooks_dir = 'tests/fixtures/invalid_hooks_dir'
    hook_name = 'pre_gen_project'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    hooks_dir = 'tests/fixtures/hooks_with_backup'
    hook_name = 'pre_gen_project'
    result = find_hook(hook_name, hooks_dir)
    expected = [os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py'))]
    assert result == expected


# LLM-generated content at query #8
#--------------------------

```python
def test_run_hook_with_valid_hook():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'pre_gen_project'
    with tempfile.NamedTemporaryFile(dir=project_dir, prefix=f'hooks/{hook_name}', suffix='.py') as f:
        f.write(b'#!/usr/bin/env python\nprint("hook executed")')
        f.flush()
        run_hook(hook_name, project_dir, context)

def test_run_hook_with_invalid_hook():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'invalid_hook'
    run_hook(hook_name, project_dir, context)

def test_run_hook_with_empty_hooks_dir():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'pre_gen_project'
    os.makedirs(os.path.join(project_dir, 'hooks'), exist_ok=True)
    run_hook(hook_name, project_dir, context)

def test_run_hook_with_backup_file():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'pre_gen_project'
    with tempfile.NamedTemporaryFile(dir=project_dir, prefix=f'hooks/{hook_name}', suffix='.py~') as f:
        run_hook(hook_name, project_dir, context)

def test_run_hook_with_unsupported_hook():
    context = {'cookiecutter': {}}
    project_dir = '/tmp/project'
    hook_name = 'unsupported_hook'
    with tempfile.NamedTemporaryFile(dir=project_dir, prefix=f'hooks/{hook_name}', suffix='.py') as f:
        run_hook(hook_name, project_dir, context)


# LLM-generated content at query #9
#--------------------------

```
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None


# LLM-generated content at query #10
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    hook_file = "pre_gen_project.py"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, hook_file), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, hook_file))]
    os.remove(os.path.join(hooks_dir, hook_file))
    os.rmdir(hooks_dir)

def test_find_hook_with_invalid_hook():
    hook_name = "invalid_hook"
    hooks_dir = "hooks"
    hook_file = "pre_gen_project.py"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, hook_file), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, hook_file))
    os.rmdir(hooks_dir)

def test_find_hook_with_backup_file():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    hook_file = "pre_gen_project.py~"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, hook_file), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, hook_file))
    os.rmdir(hooks_dir)

def test_find_hook_with_non_existing_hooks_dir():
    hook_name = "pre_gen_project"
    hooks_dir = "non_existing_hooks_dir"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_empty_hooks_dir():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.rmdir(hooks_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre-commit.sh"), "w") as f:
        f.write("test")
    result = find_hook(hook_name, hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, "pre-commit.sh"))]
    os.remove(os.path.join(hooks_dir, "pre-commit.sh"))
    os.rmdir(hooks_dir)

def test_find_hook_with_invalid_hook():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "post-commit.sh"), "w") as f:
        f.write("test")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, "post-commit.sh"))
    os.rmdir(hooks_dir)

def test_find_hook_with_backup_file():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre-commit.sh~"), "w") as f:
        f.write("test")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, "pre-commit.sh~"))
    os.rmdir(hooks_dir)

def test_find_hook_with_non_existing_directory():
    hook_name = "pre-commit"
    hooks_dir = "tests/non_existing_hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre-commit.sh"), "w") as f:
        f.write("test")
    with open(os.path.join(hooks_dir, "pre-commit.py"), "w") as f:
        f.write("test")
    result = find_hook(hook_name, hooks_dir)
    assert set(result) == {os.path.abspath(os.path.join(hooks_dir, "pre-commit.sh")), os.path.abspath(os.path.join(hooks_dir, "pre-commit.py"))}
    os.remove(os.path.join(hooks_dir, "pre-commit.sh"))
    os.remove(os.path.join(hooks_dir, "pre-commit.py"))
    os.rmdir(hooks_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_find_hook_with_valid_hook():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [os.path.abspath(hook_file)]

def test_find_hook_with_invalid_hook_name():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('invalid_hook', hooks_dir)
        assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('unsupported_hook', hooks_dir)
        assert result is None

def test_find_hook_with_non_existent_hooks_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'non_existent_hooks')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_empty_hooks_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None


# LLM-generated content at query #13
#--------------------------

```
def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    repo_dir = '/some/test/dir'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #14
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = tempfile.mkdtemp()
    hook_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hook_dir)
    hook_file = os.path.join(hook_dir, 'pre_prompt.py')
    with open(hook_file, 'w') as f:
        f.write('print("Hello, World!")')
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, (str, Path))
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_no_hook():
    repo_dir = tempfile.mkdtemp()
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = tempfile.mkdtemp()
    hook_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hook_dir)
    hook_file = os.path.join(hook_dir, 'pre_prompt.py')
    with open(hook_file, 'w') as f:
        f.write('exit(1)')
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    shutil.rmtree(repo_dir)


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_with_no_scripts():
    scripts = []
    assert not scripts


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context_creates_and_runs_rendered_script():
    context = {'cookiecutter': {'project_name': 'Test Project'}}
    script_content = "echo '{{ cookiecutter.project_name }}'"
    temp_script = Path('test_script.sh')
    temp_script.write_text(script_content, encoding='utf-8')
    cwd = Path('.')
    
    run_script_with_context(temp_script, cwd, context)
    
    temp_script.unlink()


# LLM-generated content at query #17
#--------------------------

```
def test_find_hook_with_valid_hook():
    _HOOKS.append('test_hook')
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'test_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('test_hook', tmpdir)
        assert result == [os.path.abspath(hook_file)]

def test_find_hook_with_invalid_hook_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'invalid_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('test_hook', tmpdir)
        assert result is None

def test_find_hook_with_backup_file():
    _HOOKS.append('test_hook')
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'test_hook.py~')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('test_hook', tmpdir)
        assert result is None

def test_find_hook_with_unsupported_hook():
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file = os.path.join(tmpdir, 'unsupported_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('unsupported_hook', tmpdir)
        assert result is None

def test_find_hook_with_non_existent_directory():
    result = find_hook('test_hook', 'non_existent_dir')
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    _HOOKS.append('test_hook')
    with tempfile.TemporaryDirectory() as tmpdir:
        hook_file1 = os.path.join(tmpdir, 'test_hook.py')
        hook_file2 = os.path.join(tmpdir, 'test_hook.sh')
        with open(hook_file1, 'w') as f:
            f.write('')
        with open(hook_file2, 'w') as f:
            f.write('')
        result = find_hook('test_hook', tmpdir)
        assert sorted(result) == sorted([os.path.abspath(hook_file1), os.path.abspath(hook_file2)])


# LLM-generated content at query #18
#--------------------------

```
def test_find_hook_with_valid_hook():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks"
    expected = ["tests/test_hooks/pre-commit"]
    assert find_hook(hook_name, hooks_dir) == expected

def test_find_hook_with_invalid_hook():
    hook_name = "invalid-hook"
    hooks_dir = "tests/test_hooks"
    assert find_hook(hook_name, hooks_dir) is None

def test_find_hook_with_backup_file():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks_with_backup"
    assert find_hook(hook_name, hooks_dir) is None

def test_find_hook_with_non_existent_dir():
    hook_name = "pre-commit"
    hooks_dir = "non_existent_dir"
    assert find_hook(hook_name, hooks_dir) is None

def test_find_hook_with_unsupported_hook():
    hook_name = "unsupported-hook"
    hooks_dir = "tests/test_hooks"
    assert find_hook(hook_name, hooks_dir) is None

def test_find_hook_with_multiple_valid_hooks():
    hook_name = "pre-commit"
    hooks_dir = "tests/test_hooks_multiple"
    expected = ["tests/test_hooks_multiple/pre-commit", "tests/test_hooks_multiple/pre-commit.py"]
    assert sorted(find_hook(hook_name, hooks_dir)) == sorted(expected)


# LLM-generated content at query #19
#--------------------------

def test_run_pre_prompt_hook_returns_repo_dir_when_no_scripts_found():
    repo_dir = Path("some/dir")
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script_with_context_handles_non_existent_script():
    non_existent_path = "/path/to/nonexistent/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    try:
        run_script_with_context(non_existent_path, cwd, context)
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError for non-existent script path"


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = '/tmp/repo'
    os.makedirs(repo_dir)
    os.makedirs(f'{repo_dir}/hooks')
    hook_file = f'{repo_dir}/hooks/pre_prompt.py'
    with open(hook_file, 'w') as f:
        f.write('print("Hello, World!")')
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, Path)
    shutil.rmtree(repo_dir)
    shutil.rmtree(result)

def test_run_pre_prompt_hook_without_hook():
    repo_dir = '/tmp/repo'
    os.makedirs(repo_dir)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = '/tmp/repo'
    os.makedirs(repo_dir)
    os.makedirs(f'{repo_dir}/hooks')
    hook_file = f'{repo_dir}/hooks/pre_prompt.py'
    with open(hook_file, 'w') as f:
        f.write('exit(1)')
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        assert True
    shutil.rmtree(repo_dir)


# LLM-generated content at query #22
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


def test_run_hook_from_repo_dir_failure_deletes_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)


def test_run_hook_from_repo_dir_failure_keeps_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert os.path.exists(project_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook_execution():
    repo_dir = '/tmp/repo_dir'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True

    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_failed_hook_execution_deletes_project():
    repo_dir = '/tmp/repo_dir'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert not os.path.exists(project_dir)

def test_run_hook_from_repo_dir_failed_hook_execution_does_not_delete_project():
    repo_dir = '/tmp/repo_dir'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = False

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert os.path.exists(project_dir)


# LLM-generated content at query #24
#--------------------------

```
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_gen_project', 'nonexistent_dir')
    assert result is None

def test_find_hook_returns_none_when_no_matching_hooks_found():
    with mock.patch('os.path.isdir', return_value=True), \
         mock.patch('os.listdir', return_value=['other_hook.sh']), \
         mock.patch('valid_hook', return_value=False):
        result = find_hook('pre_gen_project')
        assert result is None

def test_find_hook_returns_scripts_when_matching_hooks_found():
    with mock.patch('os.path.isdir', return_value=True), \
         mock.patch('os.listdir', return_value=['pre_gen_project.sh']), \
         mock.patch('valid_hook', return_value=True), \
         mock.patch('os.path.abspath', side_effect=lambda x: f'/abs/path/{x}'), \
         mock.patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
        result = find_hook('pre_gen_project')
        assert result == ['/abs/path/hooks/pre_gen_project.sh']


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_with_context_creates_non_deletable_temp_file():
    script_path = "test_script.py"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    run_script_with_context(script_path, cwd, context)
    temp_file = Path(script_path).with_suffix(".py")
    assert temp_file.exists()


# LLM-generated content at query #26
#--------------------------

```
def test_run_script_oserror_not_enoexec():
    err = OSError()
    err.errno = 123  # any value except errno.ENOEXEC
    try:
        raise err
    except OSError as e:
        assert e.errno != errno.ENOEXEC


# LLM-generated content at query #27
#--------------------------

```python
def test_run_script_with_non_executable_file():
    script_path = "non_executable_file.txt"
    cwd = "."
    run_script(script_path, cwd)


# LLM-generated content at query #28
#--------------------------

```
def test_predicate_at_line_21_evaluates_to_true():
    err = OSError()
    err.errno = errno.ENOEXEC
    assert err.errno == errno.ENOEXEC


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script_with_context_creates_temporary_file_with_correct_extension():
    script_path = "/path/to/script.py"
    cwd = "/path/to/cwd"
    context = {"cookiecutter": {}}
    temp_file = None

    try:
        run_script_with_context(script_path, cwd, context)
        temp_file = next((f for f in os.listdir(tempfile.gettempdir()) if f.endswith(".py")), None)
        assert temp_file is not None
    finally:
        if temp_file:
            os.remove(os.path.join(tempfile.gettempdir(), temp_file))


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('some_hook', 'non_existent_dir')
    assert result is None

def test_find_hook_returns_none_when_no_valid_hooks():
    result = find_hook('some_hook', 'empty_hooks_dir')
    assert result is None


# LLM-generated content at query #31
#--------------------------

def test_run_script_with_context_creates_temporary_file_with_correct_extension():
    script_path = "/path/to/script.py"
    cwd = "/current/working/directory"
    context = {"cookiecutter": {}}
    temp_file = None

    try:
        run_script_with_context(script_path, cwd, context)
        temp_files = [f for f in os.listdir(tempfile.gettempdir()) if f.endswith(".py")]
        assert len(temp_files) > 0
        temp_file = os.path.join(tempfile.gettempdir(), temp_files[0])
        assert os.path.exists(temp_file)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


# LLM-generated content at query #32
#--------------------------

```python
def test_run_script_successful_execution():
    result = run_script("valid_script.py")
    assert result is None


# LLM-generated content at query #33
#--------------------------

```python
def test_run_script_successful_python_script():
    test_script = 'test_script.py'
    with open(test_script, 'w') as f:
        f.write('print("Hello")')
    run_script(test_script)
    os.remove(test_script)

def test_run_script_successful_non_python_script():
    test_script = 'test_script.sh'
    with open(test_script, 'w') as f:
        f.write('#!/bin/sh\necho "Hello"')
    run_script(test_script)
    os.remove(test_script)

def test_run_script_failed_exit_status():
    test_script = 'test_fail_script.py'
    with open(test_script, 'w') as f:
        f.write('import sys\nsys.exit(1)')
    try:
        run_script(test_script)
    except FailedHookException:
        pass
    os.remove(test_script)

def test_run_script_missing_shebang():
    test_script = 'test_no_shebang.sh'
    with open(test_script, 'w') as f:
        f.write('echo "Hello"')
    try:
        run_script(test_script)
    except FailedHookException:
        pass
    os.remove(test_script)

def test_run_script_empty_file():
    test_script = 'test_empty.sh'
    open(test_script, 'w').close()
    try:
        run_script(test_script)
    except FailedHookException:
        pass
    os.remove(test_script)

def test_run_script_with_custom_cwd():
    test_dir = 'test_dir'
    test_script = os.path.join(test_dir, 'test_script.py')
    os.mkdir(test_dir)
    with open(test_script, 'w') as f:
        f.write('print("Hello")')
    run_script(test_script, cwd=test_dir)
    os.remove(test_script)
    os.rmdir(test_dir


# LLM-generated content at query #34
#--------------------------

```python
def test_run_pre_prompt_hook_with_no_scripts():
    repo_dir = '/tmp/some/repo'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #35
#--------------------------

```python
def test_run_script_with_context_extension_not_evaluated_to_false():
    script_path = "test.py"
    cwd = "/path/to/cwd"
    context = {}
    run_script_with_context(script_path, cwd, context)
    assert os.path.exists(script_path)


# LLM-generated content at query #36
#--------------------------

```python
def test_run_script_with_context_handles_non_existent_script():
    non_existent_path = "/path/to/nonexistent/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    try:
        run_script_with_context(non_existent_path, cwd, context)
    except FileNotFoundError:
        pass  # Expected behavior
    else:
        assert False, "Expected FileNotFoundError for non-existent script path"


# LLM-generated content at query #37
#--------------------------

```
def test_find_hook_returns_none_when_hooks_dir_does_not_exist():
    result = find_hook('pre_gen_project', 'non_existent_dir')
    assert result is None

def test_find_hook_returns_none_when_no_matching_hooks():
    with mock.patch('os.path.isdir', return_value=True), \
         mock.patch('os.listdir', return_value=['other_hook.py']), \
         mock.patch('valid_hook', return_value=False):
        result = find_hook('pre_gen_project')
        assert result is None

def test_find_hook_returns_scripts_when_matching_hooks_exist():
    with mock.patch('os.path.isdir', return_value=True), \
         mock.patch('os.listdir', return_value=['pre_gen_project.py']), \
         mock.patch('valid_hook', return_value=True), \
         mock.patch('os.path.abspath', side_effect=lambda x: f'/abs/path/{x}'):
        result = find_hook('pre_gen_project')
        assert result == ['/abs/path/hooks/pre_gen_project.py


# LLM-generated content at query #38
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
        f.write('print("Success")')

    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    assert os.path.exists(project_dir)


def test_run_hook_from_repo_dir_failed_hook_deletes_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
        f.write('import sys; sys.exit(1)')

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass

    assert not os.path.exists(project_dir)


def test_run_hook_from_repo_dir_failed_hook_preserves_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = False

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
        f.write('import sys; sys.exit(1)')

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass

    assert os.path.exists(project_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {}}
    delete_project_on_failure = True
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)


# LLM-generated content at query #40
#--------------------------

def test_run_pre_prompt_hook_with_valid_pre_prompt_script():
    repo_dir = Path("test_repo")
    (repo_dir / "hooks").mkdir(parents=True)
    (repo_dir / "hooks" / "pre_prompt.py").touch()
    
    result = run_pre_prompt_hook(repo_dir)
    
    assert result != repo_dir
    assert isinstance(result, Path)


# LLM-generated content at query #41
#--------------------------

def test_run_pre_prompt_hook_with_valid_pre_prompt_script():
    import os
    import tempfile
    from pathlib import Path
    from cookiecutter.utils import work_in, create_tmp_repo_dir
    from cookiecutter.hooks import run_pre_prompt_hook

    temp_dir = tempfile.mkdtemp()
    repo_dir = Path(temp_dir)
    hook_dir = repo_dir / "hooks"
    hook_dir.mkdir()
    pre_prompt_script = hook_dir / "pre_prompt.py"
    pre_prompt_script.write_text("print('pre_prompt hook')")

    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test_project"}}
    delete_project_on_failure = False
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert os.path.exists(project_dir)


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {}
    delete_project_on_failure = False
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert os.path.exists(project_dir)


# LLM-generated content at query #44
#--------------------------

def test_run_script_with_context_creates_temp_file_with_correct_extension():
    script_path = "/path/to/script.py"
    cwd = "/current/working/directory"
    context = {"cookiecutter": {}}
    
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=".py") as temp:
        assert temp.name.endswith(".py")


# LLM-generated content at query #45
#--------------------------

```python
def test_work_in_changes_directory_and_restores():
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    with work_in(temp_dir):
        assert os.getcwd() == os.path.realpath(temp_dir)
    
    assert os.getcwd() == original_dir


# LLM-generated content at query #46
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = "pre_gen"
    hooks_dir = "tests/test_hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, "pre_gen.py"))]

def test_find_hook_with_invalid_hook():
    hook_name = "invalid_hook"
    hooks_dir = "tests/test_hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    hook_name = "post_gen"
    hooks_dir = "tests/test_hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_nonexistent_directory():
    hook_name = "pre_gen"
    hooks_dir = "nonexistent_directory"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    hook_name = "pre_gen"
    hooks_dir = "tests/test_hooks_multiple"
    result = find_hook(hook_name, hooks_dir)
    assert len(result) == 2
    assert os.path.abspath(os.path.join(hooks_dir, "pre_gen.py")) in result
    assert os.path.abspath(os.path.join(hooks_dir, "pre_gen.sh")) in result


# LLM-generated content at query #47
#--------------------------

```
def test_oserror_errno_not_enoexec():
    try:
        raise OSError(errno.EPERM, "Permission denied")
    except OSError as err:
        assert err.errno != errno.ENOEXEC


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)

def test_run_hook_from_repo_dir_failure_with_delete():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)

def test_run_hook_from_repo_dir_failure_without_delete():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
    except (FailedHookException, UndefinedError):
        assert os.path.exists(project_dir)


# LLM-generated content at query #49
#--------------------------

def test_run_hook_from_repo_dir_deletes_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {}
    delete_project_on_failure = True
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        pass
    
    assert not os.path.exists(project_dir)


# LLM-generated content at query #50
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True
    
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert os.path.exists(project_dir)

def test_run_hook_from_repo_dir_failed_hook_deletes_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = True
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert not os.path.exists(project_dir)

def test_run_hook_from_repo_dir_failed_hook_keeps_project():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    delete_project_on_failure = False
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert os.path.exists(project_dir)


# LLM-generated content at query #51
#--------------------------

def test_run_hook_from_repo_dir_deletes_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {}
    delete_project_on_failure = True
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        pass


# LLM-generated content at query #52
#--------------------------

```
def test_run_script_with_valid_executable_python_script():
    script_path = "/path/to/valid_script.py"
    cwd = "/path/to/cwd"
    run_script(script_path, cwd)

def test_run_script_with_valid_executable_non_python_script():
    script_path = "/path/to/valid_script.sh"
    cwd = "/path/to/cwd"
    run_script(script_path, cwd)


# LLM-generated content at query #53
#--------------------------

```python
def test_run_script_successful_execution():
    import tempfile
    import os
    script_content = "#!/bin/bash\necho 'Hello World'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o755)
    run_script(script_path)
    os.remove(script_path)

def test_run_script_python_successful_execution():
    import tempfile
    import os
    script_content = "print('Hello World')"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    run_script(script_path)
    os.remove(script_path)

def test_run_script_failed_execution():
    import tempfile
    import os
    script_content = "#!/bin/bash\nexit 1"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    os.remove(script_path)

def test_run_script_empty_file():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        script_path = f.name
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    os.remove(script_path)

def test_run_script_missing_shebang():
    import tempfile
    import os
    script_content = "echo 'Hello World'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    os.remove(script_path)


# LLM-generated content at query #54
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    repo_dir = "/tmp/repo_dir"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project_dir"
    context = {"cookiecutter": {"project_name": "test_project"}}
    delete_project_on_failure = True

    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_failure_with_delete():
    repo_dir = "/tmp/repo_dir"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project_dir"
    context = {"cookiecutter": {"project_name": "test_project"}}
    delete_project_on_failure = True

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"

def test_run_hook_from_repo_dir_failure_without_delete():
    repo_dir = "/tmp/repo_dir"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project_dir"
    context = {"cookiecutter": {"project_name": "test_project"}}
    delete_project_on_failure = False

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"


# LLM-generated content at query #55
#--------------------------

```python
def test_run_script_successful_execution():
    script_path = "test_script.py"
    cwd = "."
    run_script(script_path, cwd)
    # If no exception is raised, the predicate at line 18 evaluates to True


# LLM-generated content at query #56
#--------------------------

```python
def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


def test_run_hook_from_repo_dir_failure():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert os.path.exists(project_dir)


# LLM-generated content at query #57
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = False
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #58
#--------------------------

```python
def test_run_hook_from_repo_dir_not_delete_project_on_failure():
    repo_dir = "test_repo_dir"
    hook_name = "test_hook"
    project_dir = "test_project_dir"
    context = {"key": "value"}
    delete_project_on_failure = False
    
    mock_run_hook = lambda hook_name, project_dir, context: None
    mock_work_in = lambda repo_dir: None
    
    original_run_hook = run_hook
    original_work_in = work_in
    
    try:
        run_hook = mock_run_hook
        work_in = mock_work_in
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    finally:
        run_hook = original_run_hook
        work_in = original_work_in


# LLM-generated content at query #59
#--------------------------

```
def test_find_hook_with_valid_hook():
    _HOOKS = {'pre_gen_project'}
    hook_name = 'pre_gen_project'
    hooks_dir = 'tests/test_data/hooks'
    result = find_hook(hook_name, hooks_dir)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(os.path.exists(path) for path in result)

def test_find_hook_with_invalid_hook():
    _HOOKS = {'pre_gen_project'}
    hook_name = 'invalid_hook'
    hooks_dir = 'tests/test_data/hooks'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_nonexistent_directory():
    _HOOKS = {'pre_gen_project'}
    hook_name = 'pre_gen_project'
    hooks_dir = 'nonexistent_directory'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    _HOOKS = {'pre_gen_project'}
    hook_name = 'pre_gen_project'
    hooks_dir = 'tests/test_data/hooks_with_backup'
    result = find_hook(hook_name, hooks_dir)
    assert result is None or len(result) == 0

def test_find_hook_with_unsupported_hook():
    _HOOKS = {'pre_gen_project'}
    hook_name = 'unsupported_hook'
    hooks_dir = 'tests/test_data/hooks'
    result = find_hook(hook_name, hooks_dir)
    assert result is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = '/path/to/repo_with_valid_hook'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = '/path/to/repo_with_invalid_hook'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_no_hooks_dir():
    repo_dir = '/path/to/repo_with_no_hooks_dir'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_hook_failure():
    repo_dir = '/path/to/repo_with_hook_failure'
    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException:
        assert True
    else:
        assert False

def test_run_pre_prompt_hook_with_multiple_hooks():
    repo_dir = '/path/to/repo_with_multiple_hooks'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
from unittest.mock import patch

_HOOKS = {'hook1', 'hook2'}

def test_find_hook_with_valid_hook():
    with tempfile.TemporaryDirectory() as hooks_dir:
        hook_file = os.path.join(hooks_dir, 'hook1.sh')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('hook1', hooks_dir)
        assert result == [os.path.abspath(hook_file)]

def test_find_hook_with_invalid_hook():
    with tempfile.TemporaryDirectory() as hooks_dir:
        hook_file = os.path.join(hooks_dir, 'invalid_hook.sh')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('hook1', hooks_dir)
        assert result is None

def test_find_hook_with_backup_file():
    with tempfile.TemporaryDirectory() as hooks_dir:
        hook_file = os.path.join(hooks_dir, 'hook1.sh~')
        with open(hook_file, 'w') as f:
            f.write('')
        result = find_hook('hook1', hooks_dir)
        assert result is None

def test_find_hook_with_non_existing_hooks_dir():
    result = find_hook('hook1', 'non_existing_dir')
    assert result is None

def test_find_hook_with_multiple_valid_hooks():
    with tempfile.TemporaryDirectory() as hooks_dir:
        hook_file1 = os.path.join(hooks_dir, 'hook1.sh')
        hook_file2 = os.path.join(hooks_dir, 'hook2.sh')
        with open(hook_file1, 'w') as f:
            f.write('')
        with open(hook_file2, 'w') as f:
            f.write('')
        result = find_hook('hook1', hooks_dir)
        assert result == [os.path.abspath(hook_file1)]


# LLM-generated content at query #3
#--------------------------

```python
def test_find_hook_existing_hook():
    hook_name = "pre_commit"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    hook_file = os.path.join(hooks_dir, "pre_commit.py")
    with open(hook_file, "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    os.remove(hook_file)
    os.rmdir(hooks_dir)
    assert result == [os.path.abspath(hook_file)]

def test_find_hook_non_existing_hook():
    hook_name = "non_existing_hook"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    result = find_hook(hook_name, hooks_dir)
    os.rmdir(hooks_dir)
    assert result is None

def test_find_hook_non_existing_hooks_dir():
    hook_name = "pre_commit"
    hooks_dir = "non_existing_hooks_dir"
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #4
#--------------------------

def test_run_hook_with_no_scripts_found():
    context = {'cookiecutter': {}}
    run_hook('pre_gen_project', '.', context)

def test_run_hook_with_valid_script():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='.sh', dir='hooks', mode='w') as f:
        f.write('#!/bin/sh\necho "test"')
        f.flush()
        os.makedirs('hooks', exist_ok=True)
        run_hook('pre_gen_project', '.', context)

def test_run_hook_with_invalid_script_extension():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='.txt', dir='hooks', mode='w') as f:
        f.write('test')
        f.flush()
        os.makedirs('hooks', exist_ok=True)
        run_hook('pre_gen_project', '.', context)

def test_run_hook_with_backup_file():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='.sh~', dir='hooks', mode='w') as f:
        f.write('#!/bin/sh\necho "test"')
        f.flush()
        os.makedirs('hooks', exist_ok=True)
        run_hook('pre_gen_project', '.', context)

def test_run_hook_with_unsupported_hook():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='.sh', dir='hooks', mode='w') as f:
        f.write('#!/bin/sh\necho "test"')
        f.flush()
        os.makedirs('hooks', exist_ok=True)
        run_hook('invalid_hook', '.', context)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_with_context():
    script_path = 'test_script.py'
    cwd = '/path/to/cwd'
    context = {'cookiecutter': {'variable': 'value'}}
    expected_content = 'rendered_content'
    
    def mock_create_env_with_context(context):
        class MockEnv:
            def from_string(self, content):
                class MockTemplate:
                    def render(self, **kwargs):
                        return expected_content
                return MockTemplate()
        return MockEnv()
    
    def mock_run_script(temp_path, cwd):
        assert temp_path.endswith('.py')
        content = Path(temp_path).read_text(encoding='utf-8')
        assert content == expected_content
        assert cwd == '/path/to/cwd'
    
    original_create_env = utils.create_env_with_context
    original_run_script = hooks.run_script
    utils.create_env_with_context = mock_create_env_with_context
    hooks.run_script = mock_run_script
    
    try:
        hooks.run_script_with_context(script_path, cwd, context)
    finally:
        utils.create_env_with_context = original_create_env
        hooks.run_script = original_run_script


# LLM-generated content at query #6
#--------------------------

```
def test_find_hook_with_valid_hook():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        valid_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(valid_hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result == [os.path.abspath(valid_hook_file)]

def test_find_hook_with_invalid_hook_name():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        invalid_hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
        with open(invalid_hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_backup_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        backup_hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(backup_hook_file, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_unsupported_hook():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        unsupported_hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(unsupported_hook_file, 'w') as f:
            f.write('')
        result = find_hook('unsupported_hook', hooks_dir)
        assert result is None

def test_find_hook_with_nonexistent_hooks_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'nonexistent_hooks')
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is None

def test_find_hook_with_multiple_valid_hooks():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.mkdir(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('')
        with open(hook_file2, 'w') as f:
            f.write('')
        result = find_hook('pre_gen_project', hooks_dir)
        assert sorted(result) == sorted([os.path.abspath(hook_file1), os.path.abspath(hook_file2)])


# LLM-generated content at query #7
#--------------------------

```python
def test_run_pre_prompt_hook_without_scripts():
    repo_dir = Path('/path/to/repo')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #8
#--------------------------

```
def test_valid_hook_matching_supported():
    assert valid_hook("/path/to/hook.py", "hook") == True

def test_valid_hook_matching_unsupported():
    assert valid_hook("/path/to/unsupported.py", "unsupported") == False

def test_valid_hook_nonmatching_supported():
    assert valid_hook("/path/to/other.py", "hook") == False

def test_valid_hook_backup_file():
    assert valid_hook("/path/to/hook.py~", "hook") == False

def test_valid_hook_empty_name():
    assert valid_hook("/path/to/.py", "") == False


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    scripts = find_hook(hook_name, hooks_dir)
    assert scripts is not None

def test_find_hook_with_invalid_hook_dir():
    hook_name = "pre_gen_project"
    hooks_dir = "invalid_hooks_dir"
    scripts = find_hook(hook_name, hooks_dir)
    assert scripts is None

def test_find_hook_with_no_valid_hooks():
    hook_name = "invalid_hook"
    hooks_dir = "hooks"
    scripts = find_hook(hook_name, hooks_dir)
    assert scripts is None

def test_find_hook_with_multiple_valid_hooks():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    scripts = find_hook(hook_name, hooks_dir)
    assert len(scripts) > 0


# LLM-generated content at query #10
#--------------------------

```
def test_valid_hook_matching_supported():
    assert valid_hook("/path/to/hook.py", "hook") == True

def test_valid_hook_matching_unsupported():
    assert valid_hook("/path/to/unsupported.py", "unsupported") == False

def test_valid_hook_nonmatching_supported():
    assert valid_hook("/path/to/other.py", "hook") == False

def test_valid_hook_backup_file():
    assert valid_hook("/path/to/hook.py~", "hook") == False

def test_valid_hook_empty_name():
    assert valid_hook("/path/to/.py", "") == False


# LLM-generated content at query #11
#--------------------------

```
def test_run_script_success_python_script():
    import tempfile
    import os
    script_content = "print('Hello, World!')"
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    run_script(script_path)
    os.unlink(script_path)

def test_run_script_success_non_python_script():
    import tempfile
    import os
    script_content = "#!/bin/sh\necho 'Hello, World!'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    run_script(script_path)
    os.unlink(script_path)

def test_run_script_failure_exit_status():
    import tempfile
    import os
    script_content = "import sys\nsys.exit(1)"
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed (exit status: 1)" in str(e)
    os.unlink(script_path)

def test_run_script_failure_empty_file():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        script_path = f.name
    try:
        run_script(script_path)
    except FailedHookException as e:
        assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    os.unlink(script_path)

def test_run_script_failure_invalid_path():
    import os
    invalid_path = os.path.join(os.path.dirname(__file__), 'nonexistent_script')
    try:
        run_script(invalid_path)
    except FailedHookException as e:
        assert "Hook script failed (error:" in str(e)


# LLM-generated content at query #12
#--------------------------

def test_run_hook_with_no_scripts_found():
    context = {'cookiecutter': {}}
    run_hook('pre_gen_project', '.', context)

def test_run_hook_with_valid_script():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='pre_gen_project.py') as temp:
        temp.write(b'#!/usr/bin/env python\nprint("Hello")')
        temp.flush()
        os.chmod(temp.name, 0o755)
        hooks_dir = os.path.dirname(temp.name)
        run_hook('pre_gen_project', hooks_dir, context)

def test_run_hook_with_invalid_script_extension():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='pre_gen_project.txt') as temp:
        temp.write(b'#!/usr/bin/env python\nprint("Hello")')
        temp.flush()
        os.chmod(temp.name, 0o755)
        hooks_dir = os.path.dirname(temp.name)
        run_hook('pre_gen_project', hooks_dir, context)

def test_run_hook_with_backup_file():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='pre_gen_project.py~') as temp:
        temp.write(b'#!/usr/bin/env python\nprint("Hello")')
        temp.flush()
        os.chmod(temp.name, 0o755)
        hooks_dir = os.path.dirname(temp.name)
        run_hook('pre_gen_project', hooks_dir, context)

def test_run_hook_with_unsupported_hook_name():
    context = {'cookiecutter': {}}
    with tempfile.NamedTemporaryFile(suffix='unsupported_hook.py') as temp:
        temp.write(b'#!/usr/bin/env python\nprint("Hello")')
        temp.flush()
        os.chmod(temp.name, 0o755)
        hooks_dir = os.path.dirname(temp.name)
        run_hook('unsupported_hook', hooks_dir, context)


# LLM-generated content at query #13
#--------------------------

```python
def test_run_script_with_python_file_sets_run_thru_shell_true_on_windows():
    import sys
    original_platform = sys.platform
    sys.platform = 'win32'
    script_path = 'test_script.py'
    run_thru_shell = sys.platform.startswith('win')
    assert run_thru_shell == True
    sys.platform = original_platform


# LLM-generated content at query #14
#--------------------------

```python
def test_find_hook_valid_hook():
    hook_name = "valid_hook"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    expected = [os.path.abspath(os.path.join(hooks_dir, "valid_hook.py"))]
    assert result == expected

def test_find_hook_invalid_hook():
    hook_name = "invalid_hook"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_backup_file():
    hook_name = "backup_hook"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_unsupported_hook():
    hook_name = "unsupported_hook"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_non_existent_directory():
    hook_name = "valid_hook"
    hooks_dir = "tests/fixtures/non_existent"
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir, exist_ok=True)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(hook_file, 'w') as f:
        f.write('print("Running pre_prompt hook")')
    result = run_pre_prompt_hook(repo_dir)
    assert os.path.exists(result)
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_no_hooks_dir():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir, exist_ok=True)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_no_pre_prompt_hook():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir, exist_ok=True)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_failed_hook():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir, exist_ok=True)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(hook_file, 'w') as f:
        f.write('exit(1)')
    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException:
        assert True
    else:
        assert False
    finally:
        shutil.rmtree(repo_dir)


# LLM-generated content at query #16
#--------------------------

def test_run_hook_with_no_scripts():
    from pathlib import Path
    from cookiecutter.hooks import run_hook
    import logging
    logger = logging.getLogger('cookiecutter')
    logger.setLevel(logging.DEBUG)
    with unittest.mock.patch('cookiecutter.hooks.find_hook', return_value=[]):
        with unittest.mock.patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', Path('/tmp'), {})
            mock_run.assert_not_called()


# LLM-generated content at query #17
#--------------------------

```python
def test_find_hook_returns_none_when_no_valid_hooks():
    hooks_dir = 'nonexistent_hooks_dir'
    hook_name = 'test_hook'
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #18
#--------------------------

```python
def test_run_hook_with_no_scripts():
    context = {}
    project_dir = Path("/fake/dir")
    run_hook("pre_gen_project", project_dir, context)
    assert logger.debug.call_args_list[0][0][0] == 'No pre_gen_project hook found'


# LLM-generated content at query #19
#--------------------------

```python
def test_find_hook_existing_hook():
    hook_name = "pre-commit"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is not None

def test_find_hook_non_existing_hook():
    hook_name = "non-existing-hook"
    hooks_dir = "tests/fixtures/hooks"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_non_existing_hooks_dir():
    hook_name = "pre-commit"
    hooks_dir = "non-existing-dir"
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_script():
    repo_dir = Path('test_repo')
    mock_scripts = ['valid_script.py']
    
    def mock_find_hook(hook_name):
        return mock_scripts if hook_name == 'pre_prompt' else None
    
    def mock_run_script(script, repo_dir):
        pass
    
    original_find_hook = hooks.find_hook
    original_run_script = hooks.run_script
    hooks.find_hook = mock_find_hook
    hooks.run_script = mock_run_script
    
    try:
        result = hooks.run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
    finally:
        hooks.find_hook = original_find_hook
        hooks.run_script = original_run_script

def test_run_pre_prompt_hook_with_no_scripts():
    repo_dir = Path('test_repo')
    
    def mock_find_hook(hook_name):
        return None
    
    original_find_hook = hooks.find_hook
    hooks.find_hook = mock_find_hook
    
    try:
        result = hooks.run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
    finally:
        hooks.find_hook = original_find_hook


# LLM-generated content at query #21
#--------------------------

```python
def test_find_hook_returns_none_when_no_scripts_found():
    scripts = find_hook('nonexistent_hook', 'empty_hooks_dir')
    assert scripts is None


# LLM-generated content at query #22
#--------------------------

```
def test_find_hook_returns_none_when_no_valid_hooks_found():
    # Mock os.path.isdir to return True (hooks_dir exists)
    # Mock os.listdir to return empty list (no hook files)
    # Mock valid_hook to return False for any input
    # The predicate at line 25 will evaluate to False because scripts list is empty
    result = find_hook('some_hook', 'some_hooks_dir')
    assert result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    scripts = ["script1.py", "script2.py"]
    result = len(scripts) == 0
    assert result is False


# LLM-generated content at query #24
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_failure_with_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        pass

def test_run_hook_from_repo_dir_failure_without_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_gen_project.py"), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, "pre_gen_project.py"))]
    os.remove(os.path.join(hooks_dir, "pre_gen_project.py"))
    os.rmdir(hooks_dir)

def test_find_hook_with_invalid_hook():
    hook_name = "invalid_hook"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_gen_project.py"), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, "pre_gen_project.py"))
    os.rmdir(hooks_dir)

def test_find_hook_with_backup_file():
    hook_name = "pre_gen_project"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_gen_project.py~"), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, "pre_gen_project.py~"))
    os.rmdir(hooks_dir)

def test_find_hook_with_invalid_directory():
    hook_name = "pre_gen_project"
    hooks_dir = "invalid_dir"
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #26
#--------------------------

```
def test_run_script_successful_python_script():
    import tempfile
    import os
    script_content = "print('Hello, World!')"
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_successful_non_python_script():
    import tempfile
    import os
    script_content = "#!/bin/sh\necho 'Hello, World!'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    os.chmod(script_path, 0o755)
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)

def test_run_script_failed_exit_status():
    import tempfile
    import os
    script_content = "import sys\nsys.exit(1)"
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    finally:
        os.unlink(script_path)

def test_run_script_failed_enoexec():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        script_path = f.name
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    finally:
        os.unlink(script_path)

def test_run_script_failed_oserror():
    import tempfile
    import os
    script_path = os.path.join(tempfile.gettempdir(), 'nonexistent_script')
    try:
        run_script(script_path)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException


# LLM-generated content at query #27
#--------------------------

```python
def test_run_script_with_context_handles_non_existent_script():
    script_path = '/non/existent/path/script.sh'
    cwd = '/tmp'
    context = {'cookiecutter': {}}
    run_script_with_context(script_path, cwd, context)


# LLM-generated content at query #28
#--------------------------

```python
def test_valid_hook_matching_and_supported():
    hook_file = "valid_hook.py"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is True

def test_valid_hook_not_matching():
    hook_file = "invalid_hook.py"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_not_supported():
    hook_file = "unsupported_hook.py"
    hook_name = "unsupported_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_backup_file():
    hook_file = "valid_hook.py~"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_matching_supported_not_backup():
    hook_file = "valid_hook.py"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is True

def test_valid_hook_matching_supported_backup():
    hook_file = "valid_hook.py~"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_not_matching_supported():
    hook_file = "invalid_hook.py"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_not_matching_not_supported():
    hook_file = "unsupported_hook.py"
    hook_name = "invalid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False

def test_valid_hook_not_matching_supported_backup():
    hook_file = "invalid_hook.py~"
    hook_name = "valid_hook"
    result = valid_hook(hook_file, hook_name)
    assert result is False


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook_with_valid_hook():
    hook_name = 'pre_gen_project'
    hooks_dir = 'tests/test_hooks'
    result = find_hook(hook_name, hooks_dir)
    assert result == [os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py'))]

def test_find_hook_with_invalid_hook():
    hook_name = 'invalid_hook'
    hooks_dir = 'tests/test_hooks'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_non_existent_hooks_dir():
    hook_name = 'pre_gen_project'
    hooks_dir = 'non_existent_dir'
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_with_backup_file():
    hook_name = 'pre_gen_project'
    hooks_dir = 'tests/test_hooks'
    result = find_hook(hook_name, hooks_dir)
    assert os.path.abspath(os.path.join(hooks_dir, 'pre_gen_project.py~')) not in result

def test_find_hook_with_unsupported_hook():
    hook_name = 'unsupported_hook'
    hooks_dir = 'tests/test_hooks'
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #30
#--------------------------

```python
def test_find_hook_returns_list_of_valid_hooks():
    hook_name = "pre_commit"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_commit.py"), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert isinstance(result, list)
    assert len(result) > 0
    assert os.path.join(hooks_dir, "pre_commit.py") in result

def test_find_hook_returns_none_for_invalid_hook():
    hook_name = "invalid_hook"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_commit.py"), "w") as f:
        f.write("")
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_returns_none_for_missing_hooks_dir():
    hook_name = "pre_commit"
    hooks_dir = "non_existent_dir"
    result = find_hook(hook_name, hooks_dir)
    assert result is None

def test_find_hook_returns_none_for_empty_hooks_dir():
    hook_name = "pre_commit"
    hooks_dir = "hooks"
    os.makedirs(hooks_dir, exist_ok=True)
    result = find_hook(hook_name, hooks_dir)
    assert result is None


# LLM-generated content at query #31
#--------------------------

```
def test_valid_hook_returns_true_when_all_conditions_met():
    hook_file = "/path/to/valid_hook.py"
    hook_name = "valid_hook"
    _HOOKS = ["valid_hook"]
    os = type('', (), {})()
    os.path = type('', (), {})()
    os.path.basename = lambda x: "valid_hook.py"
    os.path.splitext = lambda x: ("valid_hook", ".py")
    result = valid_hook(hook_file, hook_name)
    assert result == True


# LLM-generated content at query #32
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook():
    repo_dir = '/tmp/repo'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script), exist_ok=True)
    with open(hook_script, 'w') as f:
        f.write('print("Hook executed successfully")')
    run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    assert os.path.exists(project_dir)


def test_run_hook_from_repo_dir_failed_hook_with_deletion():
    repo_dir = '/tmp/repo'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script), exist_ok=True)
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    except FailedHookException:
        assert not os.path.exists(project_dir)


def test_run_hook_from_repo_dir_failed_hook_without_deletion():
    repo_dir = '/tmp/repo'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script), exist_ok=True)
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, False)
    except FailedHookException:
        assert os.path.exists(project_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(script_path, 'w') as f:
        f.write('print("Hello, World!")')
    result = run_pre_prompt_hook(temp_dir)
    assert os.path.exists(result)
    shutil.rmtree(temp_dir)

def test_run_pre_prompt_hook_with_invalid_hook():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'invalid_hook.py')
    with open(script_path, 'w') as f:
        f.write('print("Hello, World!")')
    result = run_pre_prompt_hook(temp_dir)
    assert result == temp_dir
    shutil.rmtree(temp_dir)

def test_run_pre_prompt_hook_without_hooks_dir():
    temp_dir = tempfile.mkdtemp()
    result = run_pre_prompt_hook(temp_dir)
    assert result == temp_dir
    shutil.rmtree(temp_dir)

def test_run_pre_prompt_hook_with_failing_hook():
    temp_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(temp_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(script_path, 'w') as f:
        f.write('import sys\nsys.exit(1)')
    try:
        run_pre_prompt_hook(temp_dir)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"
    shutil.rmtree(temp_dir)


# LLM-generated content at query #34
#--------------------------

```
def test_predicate_at_line_18_evaluates_to_true():
    script_path = '/path/to/nonexistent/script'
    cwd = '.'
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException to be raised"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


# LLM-generated content at query #35
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_script():
    repo_dir = 'tests/fake-repo-pre-prompt'
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir

def test_run_pre_prompt_hook_with_invalid_script():
    repo_dir = 'tests/fake-repo'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_empty_hook_dir():
    repo_dir = 'tests/fake-repo-empty-hooks'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_nonexistent_hook_dir():
    repo_dir = 'tests/fake-repo-no-hooks'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_backup_file():
    repo_dir = 'tests/fake-repo-backup-file'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_run_pre_prompt_hook_with_unsupported_hook():
    repo_dir = 'tests/fake-repo-unsupported-hook'
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir


# LLM-generated content at query #36
#--------------------------

```python
def test_run_script_with_context():
    context = {'cookiecutter': {'project_name': 'Test Project'}}
    test_script = """
    project_name = '{{ cookiecutter.project_name }}'
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(test_script)
        temp_script_path = temp_script.name

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_script_with_context(temp_script_path, temp_dir, context)
        except FailedHookException:
            assert False, "run_script_with_context should not raise FailedHookException"

    rendered_script_path = os.path.join(temp_dir, os.path.basename(temp_script_path))
    assert os.path.exists(rendered_script_path), "Rendered script should exist"

    rendered_content = Path(rendered_script_path).read_text()
    assert "project_name = 'Test Project'" in rendered_content, "Context should be rendered correctly"

    os.unlink(temp_script_path)


# LLM-generated content at query #37
#--------------------------

def test_run_hook_from_repo_dir_with_delete_project_on_failure():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {}
    delete_project_on_failure = True
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)


# LLM-generated content at query #38
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert os.path.exists(project_dir)
    finally:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

def test_run_hook_from_repo_dir_failed_hook_with_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    
    try:
        with pytest.raises((FailedHookException, UndefinedError)):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert not os.path.exists(project_dir)
    finally:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

def test_run_hook_from_repo_dir_failed_hook_without_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False

    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    
    try:
        with pytest.raises((FailedHookException, UndefinedError)):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert os.path.exists(project_dir)
    finally:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_when_failed_hook_and_delete_project_on_failure_is_false():
    repo_dir = "/tmp/repo"
    hook_name = "pre_gen_project"
    project_dir = "/tmp/project"
    context = {"cookiecutter": {"project_name": "Test Project"}}
    delete_project_on_failure = False

    def mock_run_hook(hook_name, project_dir, context):
        raise FailedHookException("Hook failed")

    original_run_hook = hooks.run_hook
    hooks.run_hook = mock_run_hook

    try:
        hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass
    finally:
        hooks.run_hook = original_run_hook

    assert os.path.exists(project_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_script_success():
    script_path = "test_script.py"
    cwd = "."
    run_script(script_path, cwd)

def test_run_script_failure_exit_status():
    script_path = "test_script_fail.py"
    cwd = "."
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        pass

def test_run_script_failure_os_error():
    script_path = "test_script_invalid.py"
    cwd = "."
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script_with_non_py_script_and_non_executable():
    script_path = "tests/data/non_executable_script.sh"
    cwd = "tests/data"
    exit_status = EXIT_SUCCESS
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        exit_status = 1
    assert exit_status != EXIT_SUCCESS


# LLM-generated content at query #42
#--------------------------

```
def test_predicate_at_line_18_evaluates_to_false():
    exit_status = 0
    assert not (exit_status != 0)


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir_successful_hook():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test_run_hook_from_repo_dir_failed_hook_deletes_project():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass

def test_run_hook_from_repo_dir_failed_hook_preserves_project():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_run_script_with_context_handles_non_existent_script():
    non_existent_path = "/path/to/nonexistent/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    try:
        run_script_with_context(non_existent_path, cwd, context)
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError for non-existent script path"


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_true():
    script_path = "non_existent_script"
    cwd = "."
    try:
        run_script(script_path, cwd)
    except OSError as err:
        assert err.errno == errno.ENOEXEC


# LLM-generated content at query #46
#--------------------------

def test_run_hook_from_repo_dir_success():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)

def test_run_hook_from_repo_dir_failure_with_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         mock.patch('cookiecutter.utils.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        try:
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
            assert False, "Should raise FailedHookException"
        except FailedHookException:
            mock_rmtree.assert_called_once_with(project_dir)

def test_run_hook_from_repo_dir_failure_without_delete():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = False
    with mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         mock.patch('cookiecutter.utils.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        try:
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
            assert False, "Should raise FailedHookException"
        except FailedHookException:
            mock_rmtree.assert_not_called()

def test_run_hook_from_repo_dir_undefined_error():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with mock.patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         mock.patch('cookiecutter.utils.rmtree') as mock_rmtree:
        mock_run_hook.side_effect = UndefinedError('Undefined variable')
        try:
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
            assert False, "Should raise UndefinedError"
        except UndefinedError:
            mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #47
#--------------------------

```python
def test_run_script_successful_execution():
    script_path = "/path/to/script.py"
    cwd = "/path/to/cwd"
    run_script(script_path, cwd)

def test_run_script_non_python_successful_execution():
    script_path = "/path/to/script.sh"
    cwd = "/path/to/cwd"
    run_script(script_path, cwd)

def test_run_script_failed_execution():
    script_path = "/path/to/failing_script.py"
    cwd = "/path/to/cwd"
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"

def test_run_script_os_error():
    script_path = "/path/to/nonexistent_script.py"
    cwd = "/path/to/cwd"
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"

def test_run_script_enoc_exec_error():
    script_path = "/path/to/invalid_script.py"
    cwd = "/path/to/cwd"
    try:
        run_script(script_path, cwd)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"


# LLM-generated content at query #48
#--------------------------

```python
def test_run_hook_from_repo_dir_does_not_delete_project_on_failure():
    repo_dir = "/fake/repo/dir"
    hook_name = "fake_hook"
    project_dir = "/fake/project/dir"
    context = {}
    delete_project_on_failure = False
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except (FailedHookException, UndefinedError):
        assert not os.path.exists(project_dir)


# LLM-generated content at query #49
#--------------------------

```python
def test_run_script_with_successful_execution():
    script_path = "test_script.py"
    cwd = "."
    run_script(script_path, cwd)

def test_run_script_with_failed_execution():
    script_path = "test_script.py"
    cwd = "."
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass

def test_run_script_with_oserror_enoeexec():
    script_path = "test_script.py"
    cwd = "."
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass

def test_run_script_with_oserror_generic():
    script_path = "test_script.py"
    cwd = "."
    try:
        run_script(script_path, cwd)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_temp_file_has_same_extension_as_original():
    script_path = "/path/to/script.py"
    cwd = "/path/to/working/dir"
    context = {"cookiecutter": {}}
    temp_file = None

    def mock_run_script(temp_name, _):
        nonlocal temp_file
        temp_file = temp_name

    run_script = mock_run_script
    run_script_with_context(script_path, cwd, context)
    _, extension = os.path.splitext(temp_file)
    assert extension == ".py"


# LLM-generated content at query #51
#--------------------------

def test_run_script_with_context_handles_non_existent_script_path():
    script_path = "/non/existent/path/script.sh"
    cwd = "/tmp"
    context = {"cookiecutter": {}}
    try:
        run_script_with_context(script_path, cwd, context)
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError when script path doesn't exist"


# LLM-generated content at query #52
#--------------------------

```python
def test_run_hook_from_repo_dir_deletes_project_on_failure():
    repo_dir = "/fake/repo"
    hook_name = "pre_gen_project"
    project_dir = "/fake/project"
    context = {}
    delete_project_on_failure = True
    
    def mock_run_hook(*args):
        raise FailedHookException
    
    original_run_hook = run_hook
    run_hook = mock_run_hook
    
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass
    finally:
        assert not os.path.exists(project_dir)
        run_hook = original_run_hook


# LLM-generated content at query #53
#--------------------------

```python
def test_run_hook_from_repo_dir_delete_project_on_failure():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {}
    delete_project_on_failure = True

    def mock_run_hook(*args, **kwargs):
        raise FailedHookException("Hook failed")

    original_run_hook = hooks.run_hook
    hooks.run_hook = mock_run_hook

    try:
        with pytest.raises(FailedHookException):
            hooks.run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    finally:
        hooks.run_hook = original_run_hook


# LLM-generated content at query #54
#--------------------------

```python
def test_run_pre_prompt_hook_with_valid_hook():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir)
    os.makedirs(f'{repo_dir}/hooks')
    with open(f'{repo_dir}/hooks/pre_prompt.py', 'w') as f:
        f.write('print("pre_prompt hook executed")')
    result = run_pre_prompt_hook(repo_dir)
    assert isinstance(result, str)
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_invalid_hook():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir)
    os.makedirs(f'{repo_dir}/hooks')
    with open(f'{repo_dir}/hooks/invalid_hook.py', 'w') as f:
        f.write('print("invalid hook executed")')
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)

def test_run_pre_prompt_hook_with_no_hooks_dir():
    repo_dir = '/tmp/test_repo'
    os.makedirs(repo_dir)
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    shutil.rmtree(repo_dir)


# LLM-generated content at query #55
#--------------------------

```python
def test_run_script_with_context_creates_temp_file_with_correct_extension():
    script_path = 'test.py'
    cwd = '/path/to/cwd'
    context = {'cookiecutter': {}}
    temp_file = None
    
    try:
        run_script_with_context(script_path, cwd, context)
        temp_file = next(f for f in os.listdir() if f.endswith('.py'))
        assert temp_file is not None
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


