####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja2 template syntax
    script_file = tmp_path / "test_script.py"
    script_content = """
import os
with open('{{ output_file }}', 'w') as f:
    f.write('{{ greeting }} {{ name }}')
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    # Create context for rendering
    context = {
        'output_file': str(tmp_path / 'output.txt'),
        'greeting': 'Hello',
        'name': 'World'
    }
    
    # Run the script with context
    run_script_with_context(script_file, tmp_path, context)
    
    # Verify the script was executed and output file was created
    output_file = tmp_path / 'output.txt'
    assert output_file.exists()
    assert output_file.read_text() == 'Hello World'


def test_run_script_with_context_bash(tmp_path, monkeypatch):
    """Test run_script_with_context with bash script."""
    script_file = tmp_path / "test_script.sh"
    script_content = """#!/bin/bash
echo "{{ message }}" > {{ output_file }}
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'message': 'Test message',
        'output_file': str(tmp_path / 'output.txt')
    }
    
    run_script_with_context(script_file, tmp_path, context)
    
    output_file = tmp_path / 'output.txt'
    assert output_file.exists()
    assert 'Test message' in output_file.read_text()


def test_run_script_with_context_undefined_variable(tmp_path):
    """Test run_script_with_context raises UndefinedError for missing context variables."""
    script_file = tmp_path / "test_script.py"
    script_content = """
print('{{ undefined_var }}')
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {}
    
    with pytest.raises(UndefinedError):
        run_script_with_context(script_file, tmp_path, context)


def test_run_script_with_context_with_cwd(tmp_path):
    """Test run_script_with_context executes in specified working directory."""
    script_file = tmp_path / "test_script.py"
    script_content = """
import os
with open('result.txt', 'w') as f:
    f.write(os.getcwd())
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    
    context = {}
    
    run_script_with_context(script_file, work_dir, context)
    
    result_file = work_dir / 'result.txt'
    assert result_file.exists()
    assert str(work_dir) in result_file.read_text()


def test_run_script_with_context_complex_template(tmp_path):
    """Test run_script_with_context with complex Jinja2 template."""
    script_file = tmp_path / "test_script.py"
    script_content = """
data = {
    'name': '{{ project_name }}',
    'version': '{{ version }}',
    'items': [{% for item in items %}'{{ item }}'{{ ", " if not loop.last else "" }}{% endfor %}]
}
with open('{{ output_file }}', 'w') as f:
    f.write(str(data))
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'project_name': 'MyProject',
        'version': '1.0.0',
        'items': ['item1', 'item2', 'item3'],
        'output_file': str(tmp_path / 'output.txt')
    }
    
    run_script_with_context(script_file, tmp_path, context)
    
    output_file = tmp_path / 'output.txt'
    assert output_file.exists()
    content = output_file.read_text()
    assert 'MyProject' in content
    assert '1.0.0' in content


# LLM-generated content at query #2
#--------------------------

```python
def test_find_hook(tmp_path):
    """Test find_hook function."""
    # Test: No hooks directory
    hooks_dir = tmp_path / "hooks"
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is None
    finally:
        os.chdir(original_cwd)

    # Test: Empty hooks directory
    hooks_dir.mkdir()
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is None
    finally:
        os.chdir(original_cwd)

    # Test: Valid hook file found
    hook_file = hooks_dir / "pre_gen_project.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('test')")
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file)
    finally:
        os.chdir(original_cwd)

    # Test: Multiple valid hook files
    hook_file_2 = hooks_dir / "pre_gen_project.sh"
    hook_file_2.write_text("#!/bin/bash\necho 'test'")
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is not None
        assert len(result) == 2
    finally:
        os.chdir(original_cwd)

    # Test: Backup files are ignored
    backup_file = hooks_dir / "pre_gen_project.py~"
    backup_file.write_text("#!/usr/bin/env python\nprint('backup')")
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is not None
        assert len(result) == 2
        assert str(backup_file) not in result
    finally:
        os.chdir(original_cwd)

    # Test: Unsupported hook names are ignored
    unsupported_file = hooks_dir / "unsupported_hook.py"
    unsupported_file.write_text("#!/usr/bin/env python\nprint('unsupported')")
    try:
        os.chdir(tmp_path)
        result = find_hook("pre_gen_project", "hooks")
        assert result is not None
        assert len(result) == 2
        assert str(unsupported_file) not in result
    finally:
        os.chdir(original_cwd)

    # Test: Different hook name
    post_gen_file = hooks_dir / "post_gen_project.py"
    post_gen_file.write_text("#!/usr/bin/env python\nprint('post')")
    try:
        os.chdir(tmp_path)
        result = find_hook("post_gen_project", "hooks")
        assert result is not None
        assert len(result) == 1
        assert str(post_gen_file) in result
        assert str(hook_file) not in result
    finally:
        os.chdir(original_cwd)

    # Test: Non-existent hook
    try:
        os.chdir(tmp_path)
        result = find_hook("non_existent_hook", "hooks")
        assert result is None
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #3
#--------------------------

```python
def test_run_hook_from_repo_dir(mocker, tmp_path):
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test hook failure with delete_project_on_failure=True
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test hook failure with delete_project_on_failure=False
    mock_rmtree.reset_mock()
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError with delete_project_on_failure=True
    mock_rmtree.reset_mock()
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    # Setup directories
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    context = {'key': 'value'}
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    assert project_dir.exists()
    
    # Test failed hook execution with cleanup
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    
    # Test failed hook execution without cleanup
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError handling with cleanup
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test case 1: No pre_prompt hook found
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test case 2: pre_prompt hook exists and succeeds
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    hook_script = hooks_dir2 / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\nexit 0\n")
    hook_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert os.path.isdir(result)
    
    # Test case 3: pre_prompt hook fails
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    hook_script3 = hooks_dir3 / "pre_prompt.sh"
    hook_script3.write_text("#!/bin/bash\nexit 1\n")
    hook_script3.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test case 4: pre_prompt hook with Python script
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    hook_script4 = hooks_dir4 / "pre_prompt.py"
    hook_script4.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)\n")
    hook_script4.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert os.path.isdir(result)
    
    # Test case 5: Multiple pre_prompt hooks
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    hook_script5a = hooks_dir5 / "pre_prompt.sh"
    hook_script5a.write_text("#!/bin/bash\nexit 0\n")
    hook_script5a.chmod(0o755)
    
    hook_script5b = hooks_dir5 / "pre_prompt.py"
    hook_script5b.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)\n")
    hook_script5b.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result != repo_dir5
    assert os.path.isdir(result)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    # Setup
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    assert project_dir.exists()


def test_run_hook_from_repo_dir_hook_fails_no_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir when hook fails and delete_project_on_failure is False."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    
    # Mock run_hook to raise FailedHookException
    mock_run_hook = mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed'),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    
    mock_rmtree.assert_not_called()
    assert project_dir.exists()


def test_run_hook_from_repo_dir_hook_fails_with_delete(tmp_path, mocker):
    """Test run_hook_from_repo_dir when hook fails and delete_project_on_failure is True."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    
    # Mock run_hook to raise FailedHookException
    mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=FailedHookException('Hook failed'),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_undefined_error(tmp_path, mocker):
    """Test run_hook_from_repo_dir when UndefinedError is raised."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    
    # Mock run_hook to raise UndefinedError
    mocker.patch(
        'cookiecutter.hooks.run_hook',
        side_effect=UndefinedError('Undefined variable'),
    )
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    
    mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_changes_working_directory(tmp_path, mocker):
    """Test run_hook_from_repo_dir changes working directory to repo_dir."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    original_cwd = os.getcwd()
    
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    
    # Verify we're back to the original directory
    assert os.getcwd() == original_cwd


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes a script successfully."""
    # Test successful script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    # Should not raise an exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_non_zero_exit_status(tmp_path):
    """Test run_script raises FailedHookException on non-zero exit status."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_bash(tmp_path):
    """Test run_script executes a bash script successfully."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(0o755)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_bash_failure(tmp_path):
    """Test run_script raises FailedHookException on bash script failure."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 42")
    script_file.chmod(0o755)
    
    with pytest.raises(FailedHookException, match='Hook script failed.*42'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test run_script handles ENOEXEC error for empty or shebang-less files."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("")
    
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test run_script handles OSError exceptions."""
    def mock_popen(*args, **kwargs):
        err = OSError("Permission denied")
        err.errno = errno.EACCES
        raise err
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_different_cwd(tmp_path):
    """Test run_script executes script from specified working directory."""
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    
    script_file = script_dir / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    run_script(str(script_file), cwd=str(work_dir))


def test_run_script_makes_executable(tmp_path, monkeypatch):
    """Test run_script calls make_executable on the script."""
    mock_make_executable = MagicMock()
    monkeypatch.setattr('cookiecutter.utils.make_executable', mock_make_executable)
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    run_script(str(script_file), cwd=str(tmp_path))
    mock_make_executable.assert_called_once_with(str(script_file))


# LLM-generated content at query #8
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja template
    script_content = "# {{ cookiecutter.project_name }}\necho 'Hello {{ cookiecutter.author }}'"
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content)
    
    # Create context
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'Test Author'
        }
    }
    
    # Mock run_script to verify it's called with correct rendered content
    called_scripts = []
    
    def mock_run_script(script_path, cwd):
        called_scripts.append((script_path, cwd))
        # Verify the temp script was created and contains rendered content
        assert Path(script_path).exists()
        content = Path(script_path).read_text(encoding='utf-8')
        assert 'test_project' in content
        assert 'Test Author' in content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Execute the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(called_scripts) == 1
    assert called_scripts[0][1] == str(tmp_path)


def test_run_script_with_context_python_file(tmp_path, monkeypatch):
    """Test run_script_with_context with a Python file."""
    script_content = "print('{{ cookiecutter.message }}')"
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'message': 'Success'
        }
    }
    
    called_scripts = []
    
    def mock_run_script(script_path, cwd):
        called_scripts.append(script_path)
        content = Path(script_path).read_text(encoding='utf-8')
        assert 'Success' in content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(called_scripts) == 1


def test_run_script_with_context_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context with undefined Jinja variable."""
    script_content = "echo '{{ undefined_var }}'"
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content)
    
    context = {'cookiecutter': {}}
    
    def mock_run_script(script_path, cwd):
        pass
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Should handle undefined variables gracefully (depends on Jinja env config)
    run_script_with_context(str(script_file), str(tmp_path), context)


def test_run_script_with_context_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context with complex Jinja template."""
    script_content = """#!/bin/bash
# Project: {{ cookiecutter.project_name }}
# Author: {{ cookiecutter.author }}
{% if cookiecutter.use_docker %}
echo "Docker enabled"
{% endif %}
"""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content)
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe',
            'use_docker': True
        }
    }
    
    rendered_content = []
    
    def mock_run_script(script_path, cwd):
        content = Path(script_path).read_text(encoding='utf-8')
        rendered_content.append(content)
        assert 'my_project' in content
        assert 'John Doe' in content
        assert 'Docker enabled' in content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(rendered_content) == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test that run_script_with_context preserves file extension."""
    script_content = "#!/usr/bin/env python\nprint('test')"
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content)
    
    context = {'cookiecutter': {}}
    
    created_files = []
    
    def mock_run_script(script_path, cwd):
        created_files.append(script_path)
        # Verify the temp file has .py extension
        assert script_path.endswith('.py')
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(created_files) == 1


# LLM-generated content at query #9
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Test successful hook execution
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('Hook executed')")
    
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="pre_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    assert project_dir.exists()


def test_run_hook_from_repo_dir_no_hook(tmp_path):
    """Test run_hook_from_repo_dir when no hook exists."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Should not raise when no hook exists
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="pre_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    assert project_dir.exists()


def test_run_hook_from_repo_dir_failure_no_delete(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=False."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create a failing hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="pre_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    # Project directory should still exist
    assert project_dir.exists()


def test_run_hook_from_repo_dir_failure_with_delete(tmp_path):
    """Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create a failing hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="pre_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    # Project directory should be deleted
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path):
    """Test run_hook_from_repo_dir with UndefinedError and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create a hook script with undefined variable
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('{{ undefined_var }}')")
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="pre_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    # Project directory should be deleted
    assert not project_dir.exists()


# LLM-generated content at query #10
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context executes a script with rendered context."""
    # Create a temporary script file with Jinja2 template
    script_path = tmp_path / "test_script.py"
    script_content = "print('{{ project_name }}')\n"
    script_path.write_text(script_content)
    
    # Mock run_script to verify it's called with correct path
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'project_name': 'my_project'}
    
    # Call the function
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    # Verify run_script was called once
    assert mock_run_script.call_count == 1
    
    # Get the temp file path that was passed to run_script
    called_script_path = mock_run_script.call_args[0][0]
    
    # Verify the temp file contains rendered content
    temp_file_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "print('my_project')" in temp_file_content
    
    # Verify cwd argument was passed correctly
    assert mock_run_script.call_args[0][1] == str(tmp_path)


def test_run_script_with_context_with_multiple_vars(tmp_path, mocker):
    """Test run_script_with_context with multiple context variables."""
    script_path = tmp_path / "test_script.sh"
    script_content = "echo {{ var1 }} {{ var2 }}\n"
    script_path.write_text(script_content)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'var1': 'hello', 'var2': 'world'}
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    called_script_path = mock_run_script.call_args[0][0]
    temp_file_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "echo hello world" in temp_file_content


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test run_script_with_context preserves script extension."""
    script_path = tmp_path / "test_script.py"
    script_content = "# {{ comment }}\n"
    script_path.write_text(script_content)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'comment': 'test comment'}
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    called_script_path = mock_run_script.call_args[0][0]
    assert Path(called_script_path).suffix == '.py'


def test_run_script_with_context_handles_undefined_gracefully(tmp_path, mocker):
    """Test run_script_with_context with undefined variables raises UndefinedError."""
    script_path = tmp_path / "test_script.py"
    script_content = "print('{{ undefined_var }}')\n"
    script_path.write_text(script_content)
    
    mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'other_var': 'value'}
    
    # Should raise UndefinedError when rendering undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_path), str(tmp_path), context)


def test_run_script_with_context_empty_context(tmp_path, mocker):
    """Test run_script_with_context with empty context."""
    script_path = tmp_path / "test_script.sh"
    script_content = "#!/bin/bash\necho 'test'\n"
    script_path.write_text(script_content)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), str(tmp_path), {})
    
    assert mock_run_script.call_count == 1


def test_run_script_with_context_with_pathlib_path(tmp_path, mocker):
    """Test run_script_with_context accepts pathlib.Path objects."""
    script_path = tmp_path / "test_script.py"
    script_content = "# {{ value }}\n"
    script_path.write_text(script_content)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'value': 'test'}
    
    # Pass Path objects instead of strings
    run_script_with_context(script_path, tmp_path, context)
    
    assert mock_run_script.call_count == 1


# LLM-generated content at query #11
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    hook_file = hooks_dir / "post_gen_project.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('success')")
    
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    assert project_dir.exists()
    
    # Test hook failure with delete_project_on_failure=True
    hook_file.write_text("#!/usr/bin/env python\nexit(1)")
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    
    assert not project_dir.exists()
    
    # Test hook failure with delete_project_on_failure=False
    project_dir.mkdir()
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    
    assert project_dir.exists()


def test_run_hook_from_repo_dir_no_hook(tmp_path):
    """Test run_hook_from_repo_dir when no hook exists."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Should not raise when hook doesn't exist
    run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, False)
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path):
    """Test run_hook_from_repo_dir handles UndefinedError and cleans up."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    hook_file = hooks_dir / "post_gen_project.py"
    hook_file.write_text("#!/usr/bin/env python\nprint('{{ undefined_var }}')")
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, "post_gen_project", project_dir, context, True)
    
    assert not project_dir.exists()


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook function executes scripts correctly."""
    # Setup
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a simple Python hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("# Test hook\nprint('Hook executed')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'project_name': 'test_project',
        'author': 'test_author'
    }
    
    # Change to hooks directory to simulate find_hook behavior
    monkeypatch.chdir(tmp_path)
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify - if no exception is raised, the hook was executed successfully


def test_run_hook_no_scripts_found(tmp_path, monkeypatch, caplog):
    """Test run_hook when no hook scripts are found."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify - should log debug message and return without error
    assert 'No pre_gen_project hook found' in caplog.text


def test_run_hook_with_context_rendering(tmp_path, monkeypatch):
    """Test run_hook renders context variables in hook scripts."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create hook script with Jinja template variables
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("# Project: {{ project_name }}\nprint('{{ author }}')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'project_name': 'my_project',
        'author': 'john_doe'
    }
    
    monkeypatch.chdir(tmp_path)
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify - if no exception, context was rendered correctly


def test_run_hook_failed_execution(tmp_path, monkeypatch):
    """Test run_hook raises exception when script fails."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook script that fails
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("import sys\nsys.exit(1)")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    # Execute and verify exception is raised
    from cookiecutter.exceptions import FailedHookException
    
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook executes multiple hook scripts."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create multiple hook scripts with same name but different extensions
    hook_script1 = hooks_dir / "pre_gen_project.py"
    hook_script1.write_text("# Hook 1\nprint('First')")
    
    hook_script2 = hooks_dir / "pre_gen_project.sh"
    hook_script2.write_text("#!/bin/bash\necho 'Second'")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    # Execute - should run both scripts
    run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook function."""
    # Setup
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create a simple hook script
    hook_script = hooks_dir / 'pre_gen_project.py'
    hook_script.write_text('print("Hook executed")')
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Change to template directory
    monkeypatch.chdir(tmp_path)
    
    # Execute - should not raise
    run_hook('pre_gen_project', project_dir, context)


def test_run_hook_not_found(tmp_path, monkeypatch, caplog):
    """Test run_hook when hook is not found."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    monkeypatch.chdir(tmp_path)
    
    # Should return early without error
    run_hook('pre_gen_project', project_dir, context)
    assert 'No pre_gen_project hook found' in caplog.text


def test_run_hook_with_failed_script(tmp_path, monkeypatch):
    """Test run_hook when hook script fails."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create a hook that fails
    hook_script = hooks_dir / 'pre_gen_project.py'
    hook_script.write_text('import sys\nsys.exit(1)')
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    monkeypatch.chdir(tmp_path)
    
    # Should raise FailedHookException
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_context_rendering(tmp_path, monkeypatch):
    """Test run_hook with Jinja context rendering."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create a hook that uses context variables
    hook_script = hooks_dir / 'pre_gen_project.py'
    hook_script.write_text('# Project: {{ cookiecutter.project_name }}\nprint("OK")')
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    monkeypatch.chdir(tmp_path)
    
    # Should execute without error
    run_hook('pre_gen_project', project_dir, context)


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook with multiple hook scripts."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create multiple hook scripts
    hook_script1 = hooks_dir / 'pre_gen_project.py'
    hook_script1.write_text('print("Hook 1")')
    
    hook_script2 = hooks_dir / 'pre_gen_project.sh'
    hook_script2.write_text('#!/bin/bash\necho "Hook 2"')
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    monkeypatch.chdir(tmp_path)
    
    # Should execute both scripts
    run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #14
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found and succeeds
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    hook_script = hooks_dir2 / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\nexit 0")
    hook_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2  # Should be a temporary directory
    assert isinstance(result, (str, Path))
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    hook_script3 = hooks_dir3 / "pre_prompt.sh"
    hook_script3.write_text("#!/bin/bash\nexit 1")
    hook_script3.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt.py hook found and succeeds
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    hook_script4 = hooks_dir4 / "pre_prompt.py"
    hook_script4.write_text("import sys\nsys.exit(0)")
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert isinstance(result, (str, Path))
    
    # Test 5: No hooks directory, should return original repo_dir
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result == repo_dir5


# LLM-generated content at query #15
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes scripts correctly."""
    import sys
    
    # Test successful script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('success')")
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_non_zero_exit_status(tmp_path):
    """Test run_script raises FailedHookException on non-zero exit status."""
    import sys
    
    script_file = tmp_path / "failing_script.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(str(script_file), cwd=str(tmp_path))
    
    assert "Hook script failed (exit status: 1)" in str(exc_info.value)


def test_run_script_with_shell_script(tmp_path):
    """Test run_script executes shell scripts."""
    import sys
    
    if sys.platform.startswith('win'):
        script_file = tmp_path / "test_script.bat"
        script_file.write_text("@echo off\nexit /b 0")
    else:
        script_file = tmp_path / "test_script.sh"
        script_file.write_text("#!/bin/bash\nexit 0")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_file_not_found():
    """Test run_script raises FailedHookException when script doesn't exist."""
    with pytest.raises(FailedHookException):
        run_script("/nonexistent/path/script.py")


def test_run_script_empty_file(tmp_path):
    """Test run_script raises FailedHookException for empty file without shebang."""
    import sys
    
    if not sys.platform.startswith('win'):
        script_file = tmp_path / "empty_script.sh"
        script_file.write_text("")
        
        with pytest.raises(FailedHookException) as exc_info:
            run_script(str(script_file), cwd=str(tmp_path))
        
        assert "might be an empty file or missing a shebang" in str(exc_info.value)


def test_run_script_with_cwd_parameter(tmp_path):
    """Test run_script executes in specified working directory."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import os\nassert os.getcwd() == os.path.abspath('.')")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_python_executable(tmp_path):
    """Test run_script uses sys.executable for .py files."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('python script')")
    
    run_script(str(script_file), cwd=str(tmp_path))


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script(tmp_path, mocker):
    """Test run_script executes scripts correctly."""
    # Test successful script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_non_python(tmp_path, mocker):
    """Test run_script executes non-python scripts."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    mocker.patch('cookiecutter.utils.make_executable')
    mocker.patch('subprocess.Popen')
    mock_proc = mocker.MagicMock()
    mock_proc.wait.return_value = 0
    mocker.patch('subprocess.Popen', return_value=mock_proc)
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    assert mock_proc.wait.called


def test_run_script_with_nonzero_exit_status(tmp_path, mocker):
    """Test run_script raises FailedHookException on non-zero exit."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(str(script_file), cwd=str(tmp_path))
    
    assert 'Hook script failed (exit status: 1)' in str(exc_info.value)


def test_run_script_oserror_enoexec(tmp_path, mocker):
    """Test run_script handles OSError with ENOEXEC errno."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("invalid")
    
    mocker.patch('cookiecutter.utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_popen.side_effect = OSError(errno.ENOEXEC, 'Exec format error')
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(str(script_file), cwd=str(tmp_path))
    
    assert 'might be an empty file or missing a shebang' in str(exc_info.value)


def test_run_script_oserror_other(tmp_path, mocker):
    """Test run_script handles other OSError exceptions."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("test")
    
    mocker.patch('cookiecutter.utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_popen.side_effect = OSError(errno.EACCES, 'Permission denied')
    
    with pytest.raises(FailedHookException) as exc_info:
        run_script(str(script_file), cwd=str(tmp_path))
    
    assert 'Hook script failed (error:' in str(exc_info.value)


def test_run_script_python_executable(tmp_path, mocker):
    """Test run_script uses sys.executable for Python scripts."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    mocker.patch('cookiecutter.utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.MagicMock()
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    call_args = mock_popen.call_args
    assert call_args[0][0][0] == sys.executable
    assert call_args[0][0][1] == str(script_file)


def test_run_script_windows_shell(tmp_path, mocker):
    """Test run_script uses shell on Windows."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("test")
    
    mocker.patch('cookiecutter.utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.MagicMock()
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc
    mocker.patch('sys.platform', 'win32')
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    call_args = mock_popen.call_args
    assert call_args[1]['shell'] is True


def test_run_script_cwd_parameter(tmp_path, mocker):
    """Test run_script passes cwd parameter to subprocess."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("test")
    cwd_dir = tmp_path / "workdir"
    cwd_dir.mkdir()
    
    mocker.patch('cookiecutter.utils.make_executable')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.MagicMock()
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc
    
    run_script(str(script_file), cwd=str(cwd_dir))
    
    call_args = mock_popen.call_args
    assert call_args[1]['cwd'] == str(cwd_dir)


# LLM-generated content at query #17
#--------------------------

```python
import os
import tempfile
from pathlib import Path

import pytest


def test_find_hook():
    """Test find_hook function with various scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Test 1: Hook file found
        hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
        Path(hook_file).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert result[0] == os.path.abspath(hook_file)
        
        # Test 2: Hook file not found
        result = find_hook('non_existent', hooks_dir)
        assert result is None
        
        # Test 3: Multiple hook files with same name
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.py')
        Path(hook_file2).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        
        # Test 4: Backup files should be ignored
        backup_file = os.path.join(hooks_dir, 'pre_prompt.sh~')
        Path(backup_file).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert all(not f.endswith('~') for f in result)
        
        # Test 5: Unsupported hook name
        unsupported_file = os.path.join(hooks_dir, 'unsupported_hook.sh')
        Path(unsupported_file).touch()
        result = find_hook('unsupported_hook', hooks_dir)
        assert result is None
        
        # Test 6: Non-existent hooks directory
        result = find_hook('pre_prompt', os.path.join(tmpdir, 'non_existent'))
        assert result is None


def test_find_hook_supported_hooks():
    """Test find_hook with all supported hook names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        supported_hooks = ['pre_prompt', 'pre_gen_project', 'post_gen_project']
        
        for hook_name in supported_hooks:
            hook_file = os.path.join(hooks_dir, f'{hook_name}.sh')
            Path(hook_file).touch()
            result = find_hook(hook_name, hooks_dir)
            assert result is not None
            assert len(result) == 1
            assert result[0] == os.path.abspath(hook_file)


def test_find_hook_with_default_hooks_dir():
    """Test find_hook with default hooks directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            hooks_dir = os.path.join(tmpdir, 'hooks')
            os.makedirs(hooks_dir)
            
            hook_file = os.path.join(hooks_dir, 'pre_prompt.sh')
            Path(hook_file).touch()
            
            result = find_hook('pre_prompt', 'hooks')
            assert result is not None
            assert len(result) == 1
        finally:
            os.chdir(original_cwd)


def test_find_hook_empty_hooks_directory():
    """Test find_hook with empty hooks directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None


def test_find_hook_multiple_extensions():
    """Test find_hook with hooks having different extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        extensions = ['.sh', '.py', '.bat', '.ps1']
        for ext in extensions:
            hook_file = os.path.join(hooks_dir, f'pre_gen_project{ext}')
            Path(hook_file).touch()
        
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 4
        for script in result:
            assert os.path.basename(script).startswith('pre_gen_project')


# LLM-generated content at query #18
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja template syntax
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    # Create context with template variables
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Mock run_script to verify it's called with correct path
    run_script_called = []
    
    def mock_run_script(path, cwd):
        run_script_called.append((path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(run_script_called) == 1
    assert run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_python_script(tmp_path, monkeypatch):
    """Test run_script_with_context with a Python script."""
    script_content = 'print("{{ variable }}")'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'variable': 'hello_world'}
    
    run_script_called = []
    
    def mock_run_script(path, cwd):
        run_script_called.append((path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(run_script_called) == 1
    assert run_script_called[0][1] == str(tmp_path)


def test_run_script_with_context_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context with undefined variable raises UndefinedError."""
    script_content = 'echo "{{ undefined_var }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    # Don't mock run_script - let it fail naturally with undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_path), str(tmp_path), context)


def test_run_script_with_context_temp_file_created(tmp_path, monkeypatch):
    """Test run_script_with_context creates temporary file with correct extension."""
    script_content = '#!/bin/bash\necho "{{ name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'name': 'test_name'}
    
    temp_files_created = []
    
    def mock_run_script(path, cwd):
        # Verify temp file exists and has correct extension
        assert os.path.exists(path)
        assert path.endswith('.sh')
        temp_files_created.append(path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert len(temp_files_created) == 1


def test_run_script_with_context_with_path_object(tmp_path, monkeypatch):
    """Test run_script_with_context accepts Path objects."""
    script_content = 'echo "{{ value }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'value': 'test_value'}
    
    run_script_called = []
    
    def mock_run_script(path, cwd):
        run_script_called.append((path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Pass Path objects
    run_script_with_context(script_path, tmp_path, context)
    
    assert len(run_script_called) == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook exists and succeeds
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_path = hooks_dir2 / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\nexit 0")
    script_path.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert isinstance(result, (str, Path))
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    script_path3 = hooks_dir3 / "pre_prompt.sh"
    script_path3.write_text("#!/bin/bash\nexit 1")
    script_path3.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt hook with .py extension
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    script_path4 = hooks_dir4 / "pre_prompt.py"
    script_path4.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    script_path4.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    
    # Test 5: Multiple pre_prompt hooks
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    script_path5a = hooks_dir5 / "pre_prompt.sh"
    script_path5a.write_text("#!/bin/bash\nexit 0")
    script_path5a.chmod(0o755)
    
    script_path5b = hooks_dir5 / "pre_prompt.py"
    script_path5b.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    script_path5b.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result != repo_dir5


# LLM-generated content at query #20
#--------------------------

```python
def test_valid_hook():
    """Test the valid_hook function with various inputs."""
    # Test valid hook file
    assert valid_hook('pre_prompt.py', 'pre_prompt') is True
    assert valid_hook('pre_prompt.sh', 'pre_prompt') is True
    assert valid_hook('pre_gen_project.py', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.sh', 'post_gen_project') is True

    # Test backup files (should be invalid)
    assert valid_hook('pre_prompt.py~', 'pre_prompt') is False
    assert valid_hook('pre_gen_project.sh~', 'pre_gen_project') is False

    # Test mismatched hook names
    assert valid_hook('pre_prompt.py', 'post_gen_project') is False
    assert valid_hook('pre_gen_project.sh', 'pre_prompt') is False

    # Test unsupported hook names
    assert valid_hook('unsupported_hook.py', 'unsupported_hook') is False
    assert valid_hook('invalid.sh', 'invalid') is False

    # Test files with different extensions
    assert valid_hook('pre_prompt.txt', 'pre_prompt') is True
    assert valid_hook('pre_prompt', 'pre_prompt') is True

    # Test with path separators in filename
    assert valid_hook('/path/to/pre_prompt.py', 'pre_prompt') is True
    assert valid_hook('subdir/pre_gen_project.sh', 'pre_gen_project') is True

    # Test edge cases
    assert valid_hook('pre_prompt.py.bak', 'pre_prompt') is False
    assert valid_hook('pre_prompt_old.py', 'pre_prompt') is False
    assert valid_hook('pre_prompter.py', 'pre_prompt') is False

    # Test with multiple extensions
    assert valid_hook('pre_prompt.py.py', 'pre_prompt') is True
    assert valid_hook('pre_prompt.tar.gz', 'pre_prompt') is True


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found and executes successfully
    repo_dir_with_hook = tmp_path / "template_with_hook"
    repo_dir_with_hook.mkdir()
    hooks_dir_with_hook = repo_dir_with_hook / "hooks"
    hooks_dir_with_hook.mkdir()
    
    hook_script = hooks_dir_with_hook / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\necho 'test'")
    hook_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir_with_hook)
    assert result != repo_dir_with_hook
    assert os.path.exists(result)
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir_fail = tmp_path / "template_fail"
    repo_dir_fail.mkdir()
    hooks_dir_fail = repo_dir_fail / "hooks"
    hooks_dir_fail.mkdir()
    
    hook_script_fail = hooks_dir_fail / "pre_prompt.sh"
    hook_script_fail.write_text("#!/bin/bash\nexit 1")
    hook_script_fail.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir_fail)
    
    # Test 4: Python pre_prompt hook
    repo_dir_py = tmp_path / "template_py"
    repo_dir_py.mkdir()
    hooks_dir_py = repo_dir_py / "hooks"
    hooks_dir_py.mkdir()
    
    hook_script_py = hooks_dir_py / "pre_prompt.py"
    hook_script_py.write_text("import sys\nprint('test')\nsys.exit(0)")
    
    result = run_pre_prompt_hook(repo_dir_py)
    assert result != repo_dir_py
    assert os.path.exists(result)
    
    # Test 5: Multiple pre_prompt hooks
    repo_dir_multi = tmp_path / "template_multi"
    repo_dir_multi.mkdir()
    hooks_dir_multi = repo_dir_multi / "hooks"
    hooks_dir_multi.mkdir()
    
    hook_script_1 = hooks_dir_multi / "pre_prompt.sh"
    hook_script_1.write_text("#!/bin/bash\necho 'test1'")
    hook_script_1.chmod(0o755)
    
    hook_script_2 = hooks_dir_multi / "pre_prompt.py"
    hook_script_2.write_text("print('test2')")
    
    result = run_pre_prompt_hook(repo_dir_multi)
    assert result != repo_dir_multi
    assert os.path.exists(result)


# LLM-generated content at query #22
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: No hooks directory exists
    result = find_hook('pre_prompt', 'hooks')
    assert result is None
    
    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None
    
    # Test 3: Hook file exists with matching name
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())
    
    # Test 4: Multiple hook files with same name but different extensions
    hook_file_py = hooks_dir / 'pre_prompt.py'
    hook_file_py.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    
    # Test 5: Backup file should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.absolute()) not in result
    
    # Test 6: Non-matching hook name should not be returned
    other_hook = hooks_dir / 'post_gen_project.sh'
    other_hook.write_text('#!/bin/bash\necho "other"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(other_hook.absolute()) not in result
    
    # Test 7: Non-existent hook name returns None
    result = find_hook('nonexistent_hook', str(hooks_dir))
    assert result is None
    
    # Test 8: Unsupported hook name returns None
    unsupported_hook = hooks_dir / 'unsupported_hook.sh'
    unsupported_hook.write_text('#!/bin/bash\necho "unsupported"')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None
    
    # Test 9: All valid hook names
    for hook_name in ['pre_prompt', 'pre_gen_project', 'post_gen_project']:
        hook_file = hooks_dir / f'{hook_name}.sh'
        hook_file.write_text('#!/bin/bash\necho "test"')
        result = find_hook(hook_name, str(hooks_dir))
        assert result is not None
        assert len(result) >= 1


# LLM-generated content at query #23
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test case 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test case 2: pre_prompt hook found and executes successfully
    repo_dir2 = tmp_path / "template2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_path = hooks_dir2 / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert os.path.isdir(result)
    
    # Test case 3: pre_prompt hook fails
    repo_dir3 = tmp_path / "template3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    script_path3 = hooks_dir3 / "pre_prompt.sh"
    script_path3.write_text("#!/bin/bash\nexit 1")
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test case 4: pre_prompt Python hook
    repo_dir4 = tmp_path / "template4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    script_path4 = hooks_dir4 / "pre_prompt.py"
    script_path4.write_text("print('test')\n")
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert os.path.isdir(result)
    
    # Test case 5: Invalid hook file (backup file) should be ignored
    repo_dir5 = tmp_path / "template5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    backup_script = hooks_dir5 / "pre_prompt.sh~"
    backup_script.write_text("#!/bin/bash\nexit 1")
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result == repo_dir5


# LLM-generated content at query #24
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Test 1: No hooks directory exists
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_prompt', 'hooks')
    assert result is None

    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None

    # Test 3: Hook file with matching name found
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())

    # Test 4: Multiple hook files with same name
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file2.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2

    # Test 5: Backup files should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.absolute()) not in result

    # Test 6: Non-matching hook name
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is None

    # Test 7: Unsupported hook name
    unsupported_file = hooks_dir / 'unsupported_hook.sh'
    unsupported_file.write_text('#!/bin/bash\necho "unsupported"')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

    # Test 8: Other files in hooks directory should be ignored
    other_file = hooks_dir / 'readme.txt'
    other_file.write_text('This is a readme')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert len(result) == 2

    # Test 9: Valid supported hooks
    post_gen_file = hooks_dir / 'post_gen_project.py'
    post_gen_file.write_text('print("post gen")')
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1

    # Test 10: pre_gen_project hook
    pre_gen_file = hooks_dir / 'pre_gen_project.sh'
    pre_gen_file.write_text('#!/bin/bash\necho "pre gen"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a test script with Jinja2 template syntax
    script_content = """#!/usr/bin/env python
# Test script
name = "{{ cookiecutter.project_name }}"
version = "{{ cookiecutter.version }}"
print(f"Project: {name}, Version: {version}")
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    script_path.chmod(0o755)
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'version': '1.0.0'
        }
    }
    
    # Mock subprocess.Popen to avoid actual script execution
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: mock_popen())
    
    # This should not raise an exception
    run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_with_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context raises UndefinedError for missing context."""
    script_content = """#!/usr/bin/env python
# Test script with undefined variable
name = "{{ cookiecutter.missing_var }}"
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    script_path.chmod(0o755)
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Should raise UndefinedError when rendering undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_shell_script(tmp_path, monkeypatch):
    """Test run_script_with_context with shell script."""
    script_content = """#!/bin/bash
echo "Project: {{ cookiecutter.project_name }}"
"""
    
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    script_path.chmod(0o755)
    
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: mock_popen())
    monkeypatch.setattr('cookiecutter.utils.make_executable', lambda x: None)
    
    # Should not raise an exception
    run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_hook_failure(tmp_path, monkeypatch):
    """Test run_script_with_context propagates hook execution failures."""
    script_content = """#!/usr/bin/env python
import sys
sys.exit(1)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    script_path.chmod(0o755)
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 1
    })
    
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: mock_popen())
    monkeypatch.setattr('cookiecutter.utils.make_executable', lambda x: None)
    
    # Should raise FailedHookException when script exits with non-zero status
    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_creates_temp_file(tmp_path, monkeypatch):
    """Test run_script_with_context creates temporary file with correct extension."""
    script_content = "{{ cookiecutter.name }}"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'name': 'rendered_content'}}
    
    created_files = []
    
    original_named_temp = tempfile.NamedTemporaryFile
    def mock_named_temp(*args, **kwargs):
        temp_file = original_named_temp(*args, **kwargs)
        created_files.append(temp_file.name)
        return temp_file
    
    mock_popen = type('MockPopen', (), {
        'wait': lambda self: 0
    })
    
    monkeypatch.setattr('tempfile.NamedTemporaryFile', mock_named_temp)
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: mock_popen())
    monkeypatch.setattr('cookiecutter.utils.make_executable', lambda x: None)
    
    run_script_with_context(script_path, tmp_path, context)
    
    # Verify that a temporary file was created
    assert len(created_files) > 0


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found and succeeds
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_path = hooks_dir2 / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert os.path.isdir(result)
    
    # Test 3: pre_prompt hook found but fails
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    script_path3 = hooks_dir3 / "pre_prompt.sh"
    script_path3.write_text("#!/bin/bash\nexit 1")
    
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt.py hook found and succeeds
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    script_path4 = hooks_dir4 / "pre_prompt.py"
    script_path4.write_text("#!/usr/bin/env python\nprint('test')")
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert os.path.isdir(result)
    
    # Test 5: Multiple pre_prompt hooks, all succeed
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    script_path5a = hooks_dir5 / "pre_prompt.sh"
    script_path5a.write_text("#!/bin/bash\necho 'test1'")
    
    script_path5b = hooks_dir5 / "pre_prompt.py"
    script_path5b.write_text("#!/usr/bin/env python\nprint('test2')")
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result != repo_dir5
    assert os.path.isdir(result)
    
    # Test 6: Backup file should be ignored
    repo_dir6 = tmp_path / "repo6"
    repo_dir6.mkdir()
    hooks_dir6 = repo_dir6 / "hooks"
    hooks_dir6.mkdir()
    
    script_path6 = hooks_dir6 / "pre_prompt.sh~"
    script_path6.write_text("#!/bin/bash\nexit 1")
    
    result = run_pre_prompt_hook(repo_dir6)
    assert result == repo_dir6


# LLM-generated content at query #27
#--------------------------

```python
def test_run_script(tmp_path, mocker):
    """Test run_script executes scripts correctly."""
    # Test successful script execution
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(0o755)
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_python(tmp_path):
    """Test run_script executes Python scripts correctly."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_nonzero_exit_status(tmp_path):
    """Test run_script raises FailedHookException on non-zero exit."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror_enoexec(tmp_path, mocker):
    """Test run_script handles ENOEXEC error."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("invalid script content")
    
    mock_popen = mocker.patch('subprocess.Popen')
    mock_popen.side_effect = OSError(errno.ENOEXEC, "Exec format error")
    
    with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror_other(tmp_path, mocker):
    """Test run_script handles other OSError."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    mock_popen = mocker.patch('subprocess.Popen')
    mock_popen.side_effect = OSError(errno.EACCES, "Permission denied")
    
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_cwd(tmp_path, mocker):
    """Test run_script runs from specified working directory."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(0o755)
    
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.Mock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    mock_popen.return_value = mock_proc
    
    cwd = tmp_path / "work_dir"
    run_script(str(script_file), cwd=str(cwd))
    
    mock_popen.assert_called_once()
    call_kwargs = mock_popen.call_args[1]
    assert call_kwargs['cwd'] == str(cwd)


def test_run_script_shell_on_windows(tmp_path, mocker):
    """Test run_script uses shell on Windows."""
    script_file = tmp_path / "test_script.bat"
    script_file.write_text("@echo off\nexit /b 0")
    
    mocker.patch('sys.platform', 'win32')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.Mock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    mock_popen.return_value = mock_proc
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    call_kwargs = mock_popen.call_args[1]
    assert call_kwargs['shell'] is True


def test_run_script_no_shell_on_unix(tmp_path, mocker):
    """Test run_script doesn't use shell on Unix."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(0o755)
    
    mocker.patch('sys.platform', 'linux')
    mock_popen = mocker.patch('subprocess.Popen')
    mock_proc = mocker.Mock()
    mock_proc.wait.return_value = EXIT_SUCCESS
    mock_popen.return_value = mock_proc
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    call_kwargs = mock_popen.call_args[1]
    assert call_kwargs['shell'] is False


# LLM-generated content at query #28
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script with Jinja2 template syntax
    script_path = tmp_path / "test_script.py"
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
name = "{{ cookiecutter.project_name }}"
version = "{{ cookiecutter.version }}"
with open("output.txt", "w") as f:
    f.write(f"{name}-{version}")
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    # Create context with template variables
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'version': '1.0.0'
        }
    }
    
    # Run the script with context
    run_script_with_context(script_path, tmp_path, context)
    
    # Verify the script was executed correctly
    output_file = tmp_path / "output.txt"
    assert output_file.exists()
    assert output_file.read_text() == "test_project-1.0.0"


def test_run_script_with_context_shell_script(tmp_path, monkeypatch):
    """Test run_script_with_context with shell script."""
    script_path = tmp_path / "test_script.sh"
    script_content = '''#!/bin/bash
echo "{{ cookiecutter.project_name }}" > output.txt
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'cookiecutter': {
            'project_name': 'shell_project'
        }
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    output_file = tmp_path / "output.txt"
    assert output_file.exists()
    assert "shell_project" in output_file.read_text()


def test_run_script_with_context_multiple_variables(tmp_path):
    """Test run_script_with_context with multiple context variables."""
    script_path = tmp_path / "test_script.py"
    script_content = '''#!/usr/bin/env python
result = "{{ var1 }}_{{ var2 }}_{{ var3 }}"
with open("result.txt", "w") as f:
    f.write(result)
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'var1': 'alpha',
        'var2': 'beta',
        'var3': 'gamma'
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / "result.txt"
    assert result_file.read_text() == "alpha_beta_gamma"


def test_run_script_with_context_nested_context(tmp_path):
    """Test run_script_with_context with nested context variables."""
    script_path = tmp_path / "test_script.py"
    script_content = '''#!/usr/bin/env python
data = "{{ config.name }}_{{ config.settings.debug }}"
with open("nested.txt", "w") as f:
    f.write(data)
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'config': {
            'name': 'myapp',
            'settings': {
                'debug': 'true'
            }
        }
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / "nested.txt"
    assert result_file.read_text() == "myapp_true"


def test_run_script_with_context_preserves_extension(tmp_path):
    """Test that run_script_with_context preserves file extension."""
    script_path = tmp_path / "test_script.custom"
    script_content = '''#!/usr/bin/env python
with open("custom.txt", "w") as f:
    f.write("{{ extension_test }}")
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'extension_test': 'success'}
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / "custom.txt"
    assert result_file.read_text() == "success"


def test_run_script_with_context_empty_context(tmp_path):
    """Test run_script_with_context with empty context."""
    script_path = tmp_path / "test_script.py"
    script_content = '''#!/usr/bin/env python
with open("empty.txt", "w") as f:
    f.write("no_context")
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / "empty.txt"
    assert result_file.read_text() == "no_context"


def test_run_script_with_context_special_characters(tmp_path):
    """Test run_script_with_context with special characters in context."""
    script_path = tmp_path / "test_script.py"
    script_content = '''#!/usr/bin/env python
with open("special.txt", "w") as f:
    f.write("{{ special_text }}")
'''
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'special_text': 'hello-world_123!@#'
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / "special.txt"
    assert result_file.read_text() == "hello-world_123!@#"


# LLM-generated content at query #29
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes scripts correctly."""
    # Test successful script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('success')")
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_non_zero_exit(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on non-zero exit."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_shell_script(tmp_path, monkeypatch):
    """Test run_script executes shell scripts."""
    script_file = tmp_path / "test_script.sh"
    if sys.platform.startswith('win'):
        script_file.write_text("@echo off\nexit /b 0")
    else:
        script_file.write_text("#!/bin/bash\nexit 0")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test run_script handles ENOEXEC error."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("")
    
    with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test run_script handles OSError."""
    nonexistent_script = tmp_path / "nonexistent.py"
    
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(nonexistent_script), cwd=str(tmp_path))


def test_run_script_with_cwd(tmp_path, monkeypatch):
    """Test run_script respects cwd parameter."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import os\nassert os.getcwd() == os.path.abspath('.')")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_makes_executable(tmp_path, monkeypatch):
    """Test run_script makes script executable."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    mock_make_executable = MagicMock()
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    mock_make_executable.assert_called_once_with(str(script_file))


# LLM-generated content at query #30
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found, should create temp dir and run script
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    # Create a valid pre_prompt hook script
    hook_script = hooks_dir2 / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('hook executed')")
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert Path(result).exists()
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    # Create a failing pre_prompt hook script
    hook_script3 = hooks_dir3 / "pre_prompt.py"
    hook_script3.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt hook with bash script
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    # Create a valid bash pre_prompt hook script
    hook_script4 = hooks_dir4 / "pre_prompt.sh"
    hook_script4.write_text("#!/bin/bash\necho 'bash hook executed'")
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert Path(result).exists()


# LLM-generated content at query #31
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found - should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook exists and runs successfully
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_file = hooks_dir2 / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert Path(result).exists()
    
    # Test 3: pre_prompt hook fails - should raise FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    failing_script = hooks_dir3 / "pre_prompt.sh"
    failing_script.write_text("#!/bin/bash\nexit 1")
    failing_script.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt hook with .py extension
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    py_script = hooks_dir4 / "pre_prompt.py"
    py_script.write_text("#!/usr/bin/env python\nprint('test')")
    py_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert Path(result).exists()
    
    # Test 5: Multiple pre_prompt hooks
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    script1 = hooks_dir5 / "pre_prompt.sh"
    script1.write_text("#!/bin/bash\necho 'script1'")
    script1.chmod(0o755)
    
    script2 = hooks_dir5 / "pre_prompt.py"
    script2.write_text("#!/usr/bin/env python\nprint('script2')")
    script2.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result != repo_dir5
    assert Path(result).exists()


# LLM-generated content at query #32
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: No hooks directory exists
    result = find_hook('pre_prompt', 'hooks')
    assert result is None
    
    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None
    
    # Test 3: Valid hook file found
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())
    
    # Test 4: Multiple valid hook files with same name
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file2.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    
    # Test 5: Backup files should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.absolute()) not in result
    
    # Test 6: Unsupported hook name
    unsupported_file = hooks_dir / 'unsupported_hook.sh'
    unsupported_file.write_text('#!/bin/bash\necho "unsupported"')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None
    
    # Test 7: Different hook names don't interfere
    post_gen_file = hooks_dir / 'post_gen_project.py'
    post_gen_file.write_text('print("post gen")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(post_gen_file.absolute()) not in result
    
    # Test 8: Find post_gen_project hook
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(post_gen_file.absolute())


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError


def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = '/path/to/repo'
    project_dir = '/path/to/project'
    hook_name = 'post_gen_project'
    context = {'project_name': 'test_project'}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=False)
        
        # Test successful hook execution
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name=hook_name,
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
        
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception():
    """Test run_hook_from_repo_dir cleans up project on FailedHookException."""
    repo_dir = '/path/to/repo'
    project_dir = '/path/to/project'
    hook_name = 'post_gen_project'
    context = {'project_name': 'test_project'}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
        
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error():
    """Test run_hook_from_repo_dir cleans up project on UndefinedError."""
    repo_dir = '/path/to/repo'
    project_dir = '/path/to/project'
    hook_name = 'pre_gen_project'
    context = {'project_name': 'test_project'}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=False)
        mock_run_hook.side_effect = UndefinedError('Variable undefined')
        
        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
        
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_no_cleanup_on_failure():
    """Test run_hook_from_repo_dir doesn't clean up when delete_project_on_failure is False."""
    repo_dir = '/path/to/repo'
    project_dir = '/path/to/project'
    hook_name = 'post_gen_project'
    context = {'project_name': 'test_project'}
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = Mock(return_value=None)
        mock_work_in.return_value.__exit__ = Mock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name=hook_name,
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=False
            )
        
        mock_rmtree.assert_not_called()
        mock_logger.exception.assert_called_once()


# LLM-generated content at query #34
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context executes a script with Jinja rendering."""
    # Create a temporary script file with Jinja template
    script_content = "#!/bin/bash\necho {{ cookiecutter.project_name }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    # Mock run_script to verify it's called with correct arguments
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    cwd = tmp_path
    
    # Call the function
    run_script_with_context(script_path, cwd, context)
    
    # Verify run_script was called
    assert mock_run_script.called
    call_args = mock_run_script.call_args
    temp_script_path = call_args[0][0]
    assert call_args[0][1] == cwd
    
    # Verify the temporary script contains rendered content
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'test_project' in temp_content
    assert '{{' not in temp_content


def test_run_script_with_context_py_extension(tmp_path, mocker):
    """Test run_script_with_context with Python script."""
    script_content = "print('{{ cookiecutter.name }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'name': 'my_app'}}
    cwd = tmp_path
    
    run_script_with_context(script_path, cwd, context)
    
    assert mock_run_script.called
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.py')
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'my_app' in temp_content


def test_run_script_with_context_undefined_variable(tmp_path, mocker):
    """Test run_script_with_context with undefined template variable."""
    script_content = "#!/bin/bash\necho {{ undefined_var }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'name': 'test'}}
    
    # Should raise UndefinedError when rendering undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_complex_template(tmp_path, mocker):
    """Test run_script_with_context with complex Jinja template."""
    script_content = "#!/bin/bash\n{% if cookiecutter.use_feature %}echo Feature enabled{% endif %}\necho {{ cookiecutter.version }}"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {
        'cookiecutter': {
            'use_feature': True,
            'version': '1.0.0'
        }
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    assert mock_run_script.called
    temp_script_path = mock_run_script.call_args[0][0]
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'Feature enabled' in temp_content
    assert '1.0.0' in temp_content
    assert '{%' not in temp_content


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test that run_script_with_context preserves file extension."""
    script_path = tmp_path / "test_script.bash"
    script_path.write_text("#!/bin/bash\necho {{ var }}", encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'var': 'value'}
    
    run_script_with_context(script_path, tmp_path, context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.bash')


# LLM-generated content at query #35
#--------------------------

```python
def test_run_hook(tmp_path, mocker):
    """Test run_hook function executes scripts correctly."""
    # Setup
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a mock hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("print('test hook')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'project_name': 'test_project',
        'cookiecutter': {'project_name': 'test_project'}
    }
    
    # Mock find_hook to return our test script
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = [str(hook_script)]
    
    # Mock run_script_with_context to verify it's called
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_run_script.assert_called_once_with(str(hook_script), project_dir, context)


def test_run_hook_no_scripts_found(mocker):
    """Test run_hook returns early when no scripts are found."""
    # Mock find_hook to return None
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = None
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    # Execute
    run_hook('pre_gen_project', '/some/path', {})
    
    # Verify
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_run_script.assert_not_called()


def test_run_hook_multiple_scripts(tmp_path, mocker):
    """Test run_hook executes multiple hook scripts."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create multiple hook scripts
    script1 = hooks_dir / "pre_gen_project.py"
    script2 = hooks_dir / "pre_gen_project.sh"
    script1.write_text("print('hook1')")
    script2.write_text("#!/bin/bash\necho 'hook2'")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'test': 'value'}
    
    # Mock find_hook to return multiple scripts
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = [str(script1), str(script2)]
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script_with_context')
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify both scripts were executed
    assert mock_run_script.call_count == 2
    mock_run_script.assert_any_call(str(script1), project_dir, context)
    mock_run_script.assert_any_call(str(script2), project_dir, context)


# LLM-generated content at query #36
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    import tempfile
    from pathlib import Path
    
    # Test case 1: No pre_prompt hook found, returns original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test case 2: pre_prompt hook exists and runs successfully
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    # Create a valid pre_prompt script
    script_path = hooks_dir2 / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    script_path.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2  # Should return a temporary directory
    assert isinstance(result, (Path, str))
    
    # Test case 3: pre_prompt hook fails, raises FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    # Create a failing pre_prompt script
    script_path3 = hooks_dir3 / "pre_prompt.sh"
    script_path3.write_text("#!/bin/bash\nexit 1")
    script_path3.chmod(0o755)
    
    with pytest.raises(FailedHookException):
        run_pre_prompt_hook(repo_dir3)
    
    # Test case 4: Multiple pre_prompt scripts, all execute
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    # Create multiple valid pre_prompt scripts
    script_path4a = hooks_dir4 / "pre_prompt.sh"
    script_path4a.write_text("#!/bin/bash\nexit 0")
    script_path4a.chmod(0o755)
    
    script_path4b = hooks_dir4 / "pre_prompt.py"
    script_path4b.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    script_path4b.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert isinstance(result, (Path, str))


# LLM-generated content at query #37
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Test 1: No hooks directory exists
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_gen_project')
    assert result is None

    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

    # Test 3: Hook file with matching name exists
    hook_file = hooks_dir / 'pre_gen_project.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())

    # Test 4: Multiple hook files with same name but different extensions
    hook_file_py = hooks_dir / 'pre_gen_project.py'
    hook_file_py.write_text('print("test")')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 2

    # Test 5: Hook file with backup extension should be ignored
    hooks_dir.joinpath('pre_gen_project.sh~').write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert not any(f.endswith('~') for f in result)

    # Test 6: Non-matching hook name
    result = find_hook('non_existent_hook', str(hooks_dir))
    assert result is None

    # Test 7: Unsupported hook name (not in _HOOKS)
    unsupported_hook = hooks_dir / 'unsupported_hook.sh'
    unsupported_hook.write_text('#!/bin/bash')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

    # Test 8: Multiple different valid hooks
    hooks_dir.joinpath('pre_prompt.py').write_text('print("prompt")')
    hooks_dir.joinpath('post_gen_project.sh').write_text('#!/bin/bash')
    
    result_pre_prompt = find_hook('pre_prompt', str(hooks_dir))
    assert result_pre_prompt is not None
    assert len(result_pre_prompt) == 1
    
    result_post_gen = find_hook('post_gen_project', str(hooks_dir))
    assert result_post_gen is not None
    assert len(result_post_gen) == 1

    # Test 9: Hooks directory doesn't exist with custom path
    nonexistent_dir = tmp_path / 'nonexistent'
    result = find_hook('pre_gen_project', str(nonexistent_dir))
    assert result is None


# LLM-generated content at query #38
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes scripts successfully."""
    # Test successful Python script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('Hello')")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_non_zero_exit():
    """Test run_script raises FailedHookException on non-zero exit."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import sys\nsys.exit(1)")
        f.flush()
        temp_path = f.name
    
    try:
        with pytest.raises(FailedHookException, match='Hook script failed'):
            run_script(temp_path)
    finally:
        os.unlink(temp_path)


def test_run_script_shell_script(tmp_path):
    """Test run_script executes shell scripts on non-Windows."""
    if sys.platform.startswith('win'):
        pytest.skip("Shell script test not applicable on Windows")
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(0o755)
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on ENOEXEC error."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on OSError."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_cwd(tmp_path):
    """Test run_script executes with specified working directory."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import os\nassert os.getcwd() == r'" + str(tmp_path) + "'")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_makes_executable(tmp_path, monkeypatch):
    """Test run_script calls make_executable."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    make_executable_called = []
    
    def mock_make_executable(path):
        make_executable_called.append(path)
    
    monkeypatch.setattr(utils, 'make_executable', mock_make_executable)
    
    run_script(str(script_file), cwd=str(tmp_path))
    
    assert str(script_file) in make_executable_called


# LLM-generated content at query #39
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError


def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    context = {'project_name': 'test'}
    hook_name = 'post_gen_project'
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        
        # Test successful hook execution
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
        
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)
        mock_rmtree.assert_not_called()


def test_run_hook_from_repo_dir_failed_hook_exception():
    """Test run_hook_from_repo_dir cleans up on FailedHookException."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    context = {'project_name': 'test'}
    hook_name = 'post_gen_project'
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
        
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error():
    """Test run_hook_from_repo_dir cleans up on UndefinedError."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    context = {'project_name': 'test'}
    hook_name = 'post_gen_project'
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = UndefinedError('Undefined variable')
        
        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, True)
        
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_no_cleanup_on_failure():
    """Test run_hook_from_repo_dir does not clean up when flag is False."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    context = {'project_name': 'test'}
    hook_name = 'post_gen_project'
    
    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, False)
        
        mock_rmtree.assert_not_called()
        mock_logger.exception.assert_called_once()


# LLM-generated content at query #40
#--------------------------

```python
def test_find_hook(tmp_path):
    """Test find_hook function."""
    # Test 1: No hooks directory exists
    hooks_dir = tmp_path / "hooks"
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)

    # Test 2: Hooks directory exists but is empty
    hooks_dir.mkdir()
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)

    # Test 3: Hook file exists with matching name
    hook_file = hooks_dir / "pre_gen_project.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(hook_file.absolute())
    finally:
        os.chdir(original_cwd)

    # Test 4: Multiple hook files with same name (different extensions)
    hook_file_py = hooks_dir / "pre_gen_project.py"
    hook_file_py.write_text("print('test')")
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is not None
        assert len(result) == 2
    finally:
        os.chdir(original_cwd)

    # Test 5: Backup files should be ignored
    backup_file = hooks_dir / "pre_gen_project.sh~"
    backup_file.write_text("#!/bin/bash\necho 'backup'")
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is not None
        assert len(result) == 2
        assert str(backup_file.absolute()) not in result
    finally:
        os.chdir(original_cwd)

    # Test 6: Non-matching hook names should not be returned
    other_hook = hooks_dir / "post_gen_project.sh"
    other_hook.write_text("#!/bin/bash\necho 'other'")
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_gen_project', 'hooks')
        assert result is not None
        assert len(result) == 2
        assert str(other_hook.absolute()) not in result
    finally:
        os.chdir(original_cwd)

    # Test 7: Unsupported hook names should return None
    try:
        os.chdir(tmp_path)
        result = find_hook('unsupported_hook', 'hooks')
        assert result is None
    finally:
        os.chdir(original_cwd)

    # Test 8: Custom hooks_dir parameter
    custom_hooks_dir = tmp_path / "custom_hooks"
    custom_hooks_dir.mkdir()
    custom_hook = custom_hooks_dir / "pre_prompt.sh"
    custom_hook.write_text("#!/bin/bash\necho 'custom'")
    try:
        os.chdir(tmp_path)
        result = find_hook('pre_prompt', 'custom_hooks')
        assert result is not None
        assert len(result) == 1
        assert result[0] == str(custom_hook.absolute())
    finally:
        os.chdir(original_cwd)


# LLM-generated content at query #41
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes scripts successfully."""
    # Test successful script execution
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_python(tmp_path):
    """Test run_script executes Python scripts."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_nonzero_exit(tmp_path):
    """Test run_script raises FailedHookException on non-zero exit."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_enoexec_error(tmp_path):
    """Test run_script raises FailedHookException on ENOEXEC error."""
    script_file = tmp_path / "test_script"
    script_file.write_text("")  # Empty file without shebang
    
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_oserror(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on OSError."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    def mock_popen(*args, **kwargs):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_default_cwd(tmp_path, monkeypatch):
    """Test run_script uses default cwd of '.'."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("import sys\nsys.exit(0)")
    
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        run_script(str(script_file))
    finally:
        os.chdir(original_cwd)


def test_run_script_windows_shell(tmp_path, monkeypatch):
    """Test run_script uses shell on Windows."""
    monkeypatch.setattr(sys, 'platform', 'win32')
    
    script_file = tmp_path / "test_script.bat"
    script_file.write_text("@echo off\nexit /b 0")
    
    popen_calls = []
    original_popen = subprocess.Popen
    
    def mock_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return original_popen(*args, **kwargs)
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    run_script(str(script_file), cwd=str(tmp_path))
    assert len(popen_calls) > 0
    assert popen_calls[0][1]['shell'] is True


# LLM-generated content at query #42
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir function."""
    # Setup directories
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a simple hook script
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/bash\nexit 0")
    hook_script.chmod(0o755)
    
    # Test context
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Test successful hook execution
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    # If no exception is raised, test passes
    assert project_dir.exists()


def test_run_hook_from_repo_dir_no_hooks(tmp_path):
    """Test run_hook_from_repo_dir with no hooks present."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Should not raise any exception when no hooks exist
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name="post_gen_project",
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    assert project_dir.exists()


def test_run_hook_from_repo_dir_hook_fails_delete_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=True."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook script that fails
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/bash\nexit 1")
    hook_script.chmod(0o755)
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Should raise FailedHookException and delete project directory
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    
    # Project directory should be deleted
    assert not project_dir.exists()


def test_run_hook_from_repo_dir_hook_fails_keep_project(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir with hook failure and delete_project_on_failure=False."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook script that fails
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/bash\nexit 1")
    hook_script.chmod(0o755)
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Should raise FailedHookException but keep project directory
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    
    # Project directory should still exist
    assert project_dir.exists()


def test_run_hook_from_repo_dir_undefined_error(tmp_path, monkeypatch):
    """Test run_hook_from_repo_dir with UndefinedError in template rendering."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook script with undefined template variable
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/bash\necho {{ undefined_var }}")
    hook_script.chmod(0o755)
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Should raise FailedHookException due to UndefinedError and delete project
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name="post_gen_project",
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    
    # Project directory should be deleted
    assert not project_dir.exists()


# LLM-generated content at query #43
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"project_name": "test_project"}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test FailedHookException with delete_project_on_failure=True
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test UndefinedError with delete_project_on_failure=False
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError with delete_project_on_failure=True
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='pre_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #44
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test hook failure with delete_project_on_failure=True
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test hook failure with delete_project_on_failure=False
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError handling
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #45
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook executes scripts found in hooks directory."""
    # Setup
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create a mock hook script
    hook_script = hooks_dir / 'pre_gen_project.sh'
    hook_script.write_text('#!/bin/bash\necho "test"')
    hook_script.chmod(0o755)
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Mock find_hook to return our test script
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: [str(hook_script)] if hook_name == 'pre_gen_project' else None
    )
    
    # Mock run_script_with_context to avoid actual execution
    mock_run_called = []
    def mock_run_script_with_context(script_path, cwd, context):
        mock_run_called.append((script_path, cwd, context))
    
    monkeypatch.setattr(
        'cookiecutter.hooks.run_script_with_context',
        mock_run_script_with_context
    )
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Assert
    assert len(mock_run_called) == 1
    assert mock_run_called[0][0] == str(hook_script)
    assert mock_run_called[0][1] == project_dir
    assert mock_run_called[0][2] == context


def test_run_hook_no_scripts_found(tmp_path, monkeypatch, caplog):
    """Test run_hook returns early when no scripts are found."""
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Mock find_hook to return None
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    
    mock_run_called = []
    def mock_run_script_with_context(script_path, cwd, context):
        mock_run_called.append((script_path, cwd, context))
    
    monkeypatch.setattr(
        'cookiecutter.hooks.run_script_with_context',
        mock_run_script_with_context
    )
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Assert - run_script_with_context should not be called
    assert len(mock_run_called) == 0
    assert 'No pre_gen_project hook found' in caplog.text


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook executes multiple scripts in order."""
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    
    # Create multiple hook scripts
    hook_script1 = hooks_dir / 'post_gen_project.sh'
    hook_script1.write_text('#!/bin/bash\necho "test1"')
    
    hook_script2 = hooks_dir / 'post_gen_project.py'
    hook_script2.write_text('print("test2")')
    
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    scripts = [str(hook_script1), str(hook_script2)]
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: scripts if hook_name == 'post_gen_project' else None
    )
    
    mock_run_called = []
    def mock_run_script_with_context(script_path, cwd, context):
        mock_run_called.append((script_path, cwd, context))
    
    monkeypatch.setattr(
        'cookiecutter.hooks.run_script_with_context',
        mock_run_script_with_context
    )
    
    # Execute
    run_hook('post_gen_project', project_dir, context)
    
    # Assert - both scripts should be called in order
    assert len(mock_run_called) == 2
    assert mock_run_called[0][0] == str(hook_script1)
    assert mock_run_called[1][0] == str(hook_script2)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test case 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test case 2: pre_prompt hook found, should create temp dir and run script
    repo_dir2 = tmp_path / "repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    hook_script = hooks_dir2 / "pre_prompt.py"
    hook_script.write_text("#!/usr/bin/env python\nprint('hook executed')")
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert os.path.isdir(result)
    
    # Test case 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir3 = tmp_path / "repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    hook_script3 = hooks_dir3 / "pre_prompt.py"
    hook_script3.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test case 4: pre_prompt hook with bash script
    repo_dir4 = tmp_path / "repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    hook_script4 = hooks_dir4 / "pre_prompt.sh"
    hook_script4.write_text("#!/bin/bash\necho 'hook executed'")
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert os.path.isdir(result)
    
    # Test case 5: Multiple pre_prompt hooks, all should execute
    repo_dir5 = tmp_path / "repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    hook_script5a = hooks_dir5 / "pre_prompt.py"
    hook_script5a.write_text("#!/usr/bin/env python\nprint('first hook')")
    
    hook_script5b = hooks_dir5 / "pre_prompt.sh"
    hook_script5b.write_text("#!/bin/bash\necho 'second hook'")
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result != repo_dir5
    assert os.path.isdir(result)


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
from pathlib import Path


def test_find_hook():
    """Test find_hook function with various scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        
        # Test 1: No hooks directory
        result = find_hook('pre_prompt', os.path.join(tmpdir, 'nonexistent'))
        assert result is None
        
        # Test 2: Empty hooks directory
        result = find_hook('pre_prompt', hooks_dir)
        assert result is None
        
        # Test 3: Valid hook file found
        hook_file = os.path.join(hooks_dir, 'pre_prompt.py')
        Path(hook_file).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 1
        assert os.path.basename(result[0]) == 'pre_prompt.py'
        
        # Test 4: Multiple valid hook files
        hook_file2 = os.path.join(hooks_dir, 'pre_prompt.sh')
        Path(hook_file2).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        
        # Test 5: Backup file should be ignored
        backup_file = os.path.join(hooks_dir, 'pre_prompt.py~')
        Path(backup_file).touch()
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        assert len(result) == 2
        assert backup_file not in result
        
        # Test 6: Wrong hook name
        result = find_hook('post_gen_project', hooks_dir)
        assert result is None
        
        # Test 7: Unsupported hook name
        unsupported_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        Path(unsupported_file).touch()
        result = find_hook('unsupported_hook', hooks_dir)
        assert result is None
        
        # Test 8: Valid post_gen_project hook
        post_gen_file = os.path.join(hooks_dir, 'post_gen_project.py')
        Path(post_gen_file).touch()
        result = find_hook('post_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        
        # Test 9: Valid pre_gen_project hook
        pre_gen_file = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(pre_gen_file).touch()
        result = find_hook('pre_gen_project', hooks_dir)
        assert result is not None
        assert len(result) == 1
        
        # Test 10: Absolute paths are returned
        result = find_hook('pre_prompt', hooks_dir)
        assert result is not None
        for script_path in result:
            assert os.path.isabs(script_path)


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja2 template syntax
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    cwd = str(tmp_path)
    
    # Mock run_script to verify it's called with the correct parameters
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), cwd, context)
    
    # Verify run_script was called
    assert mock_run_script.called
    call_args = mock_run_script.call_args
    temp_script_path = call_args[0][0]
    
    # Verify the temporary script contains rendered content
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'my_project' in temp_content
    assert '{{ cookiecutter.project_name }}' not in temp_content
    
    # Clean up temp file
    Path(temp_script_path).unlink()


def test_run_script_with_context_python_file(tmp_path, mocker):
    """Test run_script_with_context with a Python script."""
    script_content = 'print("{{ project_name }}")'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'test_project'}
    cwd = str(tmp_path)
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), cwd, context)
    
    assert mock_run_script.called
    temp_script_path = mock_run_script.call_args[0][0]
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'test_project' in temp_content
    
    Path(temp_script_path).unlink()


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test run_script_with_context preserves file extension."""
    script_content = '#!/usr/bin/env python3\nprint("{{ value }}")'
    script_path = tmp_path / "hook.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'value': 'test'}
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.py')
    
    Path(temp_script_path).unlink()


def test_run_script_with_context_complex_template(tmp_path, mocker):
    """Test run_script_with_context with complex Jinja2 template."""
    script_content = '''#!/bin/bash
{% for item in items %}
echo "{{ item }}"
{% endfor %}
echo "{{ config.name }}"
'''
    script_path = tmp_path / "complex_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'items': ['item1', 'item2', 'item3'],
        'config': {'name': 'myconfig'}
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    temp_content = Path(temp_script_path).read_text(encoding='utf-8')
    
    assert 'item1' in temp_content
    assert 'item2' in temp_content
    assert 'item3' in temp_content
    assert 'myconfig' in temp_content
    assert '{% for' not in temp_content
    
    Path(temp_script_path).unlink()


def test_run_script_with_context_cwd_parameter(tmp_path, mocker):
    """Test run_script_with_context passes cwd to run_script."""
    script_content = 'echo "test"'
    script_path = tmp_path / "test.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    cwd = tmp_path / "subdir"
    cwd.mkdir()
    context = {}
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_path), str(cwd), context)
    
    # Verify cwd was passed to run_script
    call_args = mock_run_script.call_args
    assert call_args[0][1] == str(cwd)
    
    Path(call_args[0][0]).unlink()


# LLM-generated content at query #4
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test hook failure without project deletion
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    # Test hook failure with project deletion
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test UndefinedError with project deletion
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    mock_rmtree.reset_mock()
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)
    
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test case 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test case 2: pre_prompt hook found, should create temp dir and run script
    repo_dir2 = tmp_path / "template2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_file = hooks_dir2 / "pre_prompt.sh"
    script_file.write_text("#!/bin/bash\necho 'test'")
    script_file.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result != repo_dir2
    assert os.path.isdir(result)
    
    # Test case 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir3 = tmp_path / "template3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    script_file3 = hooks_dir3 / "pre_prompt.sh"
    script_file3.write_text("#!/bin/bash\nexit 1")
    script_file3.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir3)
    
    # Test case 4: Python pre_prompt hook
    repo_dir4 = tmp_path / "template4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    script_file4 = hooks_dir4 / "pre_prompt.py"
    script_file4.write_text("#!/usr/bin/env python\nprint('test')")
    script_file4.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result != repo_dir4
    assert os.path.isdir(result)


# LLM-generated content at query #6
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes a script successfully."""
    import sys
    import os
    
    # Create a simple Python script
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('Hello World')\n")
    
    # Test successful execution
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_nonzero_exit_status(tmp_path):
    """Test run_script raises FailedHookException on non-zero exit status."""
    script_file = tmp_path / "failing_script.py"
    script_file.write_text("import sys\nsys.exit(1)\n")
    
    with pytest.raises(FailedHookException, match='Hook script failed \\(exit status: 1\\)'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_shell_script(tmp_path):
    """Test run_script executes shell script on non-Windows."""
    import sys
    
    if sys.platform.startswith('win'):
        pytest.skip("Shell script test skipped on Windows")
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0\n")
    
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_enoexec_error(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on ENOEXEC error."""
    import errno
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("invalid script")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test run_script raises FailedHookException on OSError."""
    import subprocess
    
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')")
    
    def mock_popen(*args, **kwargs):
        raise OSError(2, "No such file or directory")
    
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    
    with pytest.raises(FailedHookException, match='Hook script failed \\(error:'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_cwd(tmp_path):
    """Test run_script executes script from specified working directory."""
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    
    script_file = script_dir / "test_script.py"
    script_file.write_text("import os\nassert os.getcwd() == os.path.dirname(os.path.abspath(__file__))\n")
    
    run_script(str(script_file), cwd=str(script_dir))


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script_with_context(tmp_path):
    """Test run_script_with_context executes a script with Jinja rendering."""
    # Create a temporary script with Jinja template syntax
    script_content = "#!/usr/bin/env python\nprint('{{ project_name }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'my_project'}
    cwd = tmp_path
    
    # Should not raise an exception
    run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_with_bash(tmp_path):
    """Test run_script_with_context with bash script."""
    script_content = "#!/bin/bash\necho '{{ project_name }}'\n"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'test_project'}
    cwd = tmp_path
    
    run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_undefined_variable(tmp_path):
    """Test run_script_with_context with undefined Jinja variable."""
    script_content = "#!/usr/bin/env python\nprint('{{ undefined_var }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'my_project'}
    cwd = tmp_path
    
    # Should raise UndefinedError when rendering template
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_preserves_extension(tmp_path):
    """Test run_script_with_context preserves file extension."""
    script_content = "#!/usr/bin/env python\nprint('{{ name }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'name': 'test'}
    cwd = tmp_path
    
    run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_complex_template(tmp_path):
    """Test run_script_with_context with complex Jinja template."""
    script_content = """#!/usr/bin/env python
{% for item in items %}
print('{{ item }}')
{% endfor %}
"""
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'items': ['item1', 'item2', 'item3']}
    cwd = tmp_path
    
    run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_failed_execution(tmp_path):
    """Test run_script_with_context raises FailedHookException on script failure."""
    script_content = "#!/usr/bin/env python\nimport sys\nsys.exit(1)\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    cwd = tmp_path
    
    with pytest.raises(FailedHookException):
        run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_path_as_string(tmp_path):
    """Test run_script_with_context works with string path."""
    script_content = "#!/usr/bin/env python\nprint('test')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    cwd = str(tmp_path)
    
    run_script_with_context(str(script_path), cwd, context)


def test_run_script_with_context_empty_context(tmp_path):
    """Test run_script_with_context with empty context."""
    script_content = "#!/usr/bin/env python\nprint('hello')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    cwd = tmp_path
    
    run_script_with_context(script_path, cwd, context)


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from cookiecutter.exceptions import FailedHookException
from jinja2.exceptions import UndefinedError


def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    hook_name = 'post_gen_project'
    context = {'project_name': 'test'}

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        
        run_hook_from_repo_dir(
            repo_dir,
            hook_name,
            project_dir,
            context,
            delete_project_on_failure=True
        )
        
        mock_work_in.assert_called_once_with(repo_dir)
        mock_run_hook.assert_called_once_with(hook_name, project_dir, context)


def test_run_hook_from_repo_dir_failed_hook_exception():
    """Test run_hook_from_repo_dir handles FailedHookException and deletes project."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    hook_name = 'post_gen_project'
    context = {'project_name': 'test'}

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                hook_name,
                project_dir,
                context,
                delete_project_on_failure=True
            )
        
        mock_rmtree.assert_called_once_with(project_dir)
        mock_logger.exception.assert_called_once()


def test_run_hook_from_repo_dir_undefined_error():
    """Test run_hook_from_repo_dir handles UndefinedError and deletes project."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    hook_name = 'post_gen_project'
    context = {'project_name': 'test'}

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = UndefinedError('Undefined variable')
        
        with pytest.raises(UndefinedError):
            run_hook_from_repo_dir(
                repo_dir,
                hook_name,
                project_dir,
                context,
                delete_project_on_failure=True
            )
        
        mock_rmtree.assert_called_once_with(project_dir)


def test_run_hook_from_repo_dir_no_delete_on_failure():
    """Test run_hook_from_repo_dir doesn't delete project when delete_project_on_failure is False."""
    repo_dir = Path('/repo')
    project_dir = Path('/project')
    hook_name = 'post_gen_project'
    context = {'project_name': 'test'}

    with patch('cookiecutter.hooks.work_in') as mock_work_in, \
         patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
         patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
         patch('cookiecutter.hooks.logger') as mock_logger:
        
        mock_work_in.return_value.__enter__ = MagicMock()
        mock_work_in.return_value.__exit__ = MagicMock(return_value=False)
        mock_run_hook.side_effect = FailedHookException('Hook failed')
        
        with pytest.raises(FailedHookException):
            run_hook_from_repo_dir(
                repo_dir,
                hook_name,
                project_dir,
                context,
                delete_project_on_failure=False
            )
        
        mock_rmtree.assert_not_called()


# LLM-generated content at query #9
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook function executes hook scripts with context."""
    # Setup
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a simple Python hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text(
        "import os\n"
        "with open('{{ cookiecutter.output_file }}', 'w') as f:\n"
        "    f.write('{{ cookiecutter.project_name }}')\n"
    )
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'output_file': 'test_output.txt'
        }
    }
    
    # Change to hooks directory to simulate template structure
    monkeypatch.chdir(tmp_path)
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify
    output_file = project_dir / "test_output.txt"
    assert output_file.exists()
    assert output_file.read_text() == 'test_project'


def test_run_hook_no_scripts_found(tmp_path, monkeypatch, caplog):
    """Test run_hook when no hook scripts are found."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    monkeypatch.chdir(tmp_path)
    
    with caplog.at_level(logging.DEBUG):
        run_hook('pre_gen_project', project_dir, context)
    
    assert 'No pre_gen_project hook found' in caplog.text


def test_run_hook_with_failed_script(tmp_path, monkeypatch):
    """Test run_hook raises exception when hook script fails."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a hook script that fails
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("raise RuntimeError('Hook failed')\n")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_bash_script(tmp_path, monkeypatch):
    """Test run_hook with bash script."""
    if sys.platform.startswith('win'):
        pytest.skip("Bash scripts not supported on Windows")
    
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a bash hook script
    hook_script = hooks_dir / "post_gen_project.sh"
    hook_script.write_text("#!/bin/bash\necho 'hook executed' > output.txt\n")
    hook_script.chmod(0o755)
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    monkeypatch.chdir(tmp_path)
    
    run_hook('post_gen_project', project_dir, context)
    
    output_file = project_dir / "output.txt"
    assert output_file.exists()
    assert 'hook executed' in output_file.read_text()


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook executes multiple hook scripts in order."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create first hook script
    hook_script1 = hooks_dir / "pre_gen_project.py"
    hook_script1.write_text("with open('execution_order.txt', 'a') as f:\n    f.write('script1\\n')\n")
    
    # Create second hook script with different name but same hook
    hook_script2 = hooks_dir / "pre_gen_project_extra.py"
    hook_script2.write_text("# This won't match hook name\npass\n")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'cookiecutter': {}}
    
    monkeypatch.chdir(tmp_path)
    
    run_hook('pre_gen_project', project_dir, context)
    
    output_file = project_dir / "execution_order.txt"
    assert output_file.exists()
    assert 'script1' in output_file.read_text()


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook function executes scripts found by find_hook."""
    # Create a temporary hooks directory
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a simple Python hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("# Test hook\nprint('Hook executed')\n")
    hook_script.chmod(0o755)
    
    # Create a project directory
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Mock find_hook to return our test script
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: [str(hook_script)] if hook_name == 'pre_gen_project' else None
    )
    
    # Mock run_script_with_context to verify it's called
    called_scripts = []
    def mock_run_script_with_context(script_path, cwd, context):
        called_scripts.append((script_path, cwd, context))
    
    monkeypatch.setattr(
        'cookiecutter.hooks.run_script_with_context',
        mock_run_script_with_context
    )
    
    # Test context
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Call run_hook
    run_hook('pre_gen_project', str(project_dir), context)
    
    # Verify run_script_with_context was called with correct arguments
    assert len(called_scripts) == 1
    assert called_scripts[0][0] == str(hook_script)
    assert called_scripts[0][1] == str(project_dir)
    assert called_scripts[0][2] == context


def test_run_hook_no_scripts_found(monkeypatch, caplog):
    """Test run_hook when no hook scripts are found."""
    # Mock find_hook to return None
    monkeypatch.setattr('cookiecutter.hooks.find_hook', lambda hook_name: None)
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Call run_hook
    run_hook('pre_gen_project', '/tmp/project', context)
    
    # Verify debug message was logged
    assert 'No pre_gen_project hook found' in caplog.text


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook executes multiple hook scripts."""
    # Create temporary hooks directory
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create multiple hook scripts
    hook_script1 = hooks_dir / "pre_gen_project.py"
    hook_script1.write_text("# Hook 1\n")
    hook_script1.chmod(0o755)
    
    hook_script2 = hooks_dir / "pre_gen_project.sh"
    hook_script2.write_text("#!/bin/bash\n# Hook 2\n")
    hook_script2.chmod(0o755)
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Mock find_hook to return both scripts
    monkeypatch.setattr(
        'cookiecutter.hooks.find_hook',
        lambda hook_name: [str(hook_script1), str(hook_script2)] if hook_name == 'pre_gen_project' else None
    )
    
    # Track calls to run_script_with_context
    called_scripts = []
    def mock_run_script_with_context(script_path, cwd, context):
        called_scripts.append(script_path)
    
    monkeypatch.setattr(
        'cookiecutter.hooks.run_script_with_context',
        mock_run_script_with_context
    )
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Call run_hook
    run_hook('pre_gen_project', str(project_dir), context)
    
    # Verify both scripts were executed
    assert len(called_scripts) == 2
    assert str(hook_script1) in called_scripts
    assert str(hook_script2) in called_scripts


# LLM-generated content at query #11
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: No hooks directory exists
    result = find_hook('pre_prompt', 'hooks')
    assert result is None
    
    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None
    
    # Test 3: Hook file exists with matching name
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())
    
    # Test 4: Multiple valid hook files with same name
    hook_file2 = hooks_dir / 'pre_prompt.py'
    hook_file2.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    
    # Test 5: Hook file with backup extension should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.resolve()) not in result
    
    # Test 6: Unsupported hook name should return None
    result = find_hook('invalid_hook', str(hooks_dir))
    assert result is None
    
    # Test 7: File with different name should not match
    other_file = hooks_dir / 'post_gen_project.sh'
    other_file.write_text('#!/bin/bash\necho "post"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(other_file.resolve()) not in result
    
    # Test 8: Valid post_gen_project hook
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(other_file.resolve())
    
    # Test 9: Hooks directory doesn't exist (relative path)
    result = find_hook('pre_prompt', 'nonexistent_hooks')
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test_valid_hook():
    """Test the valid_hook function."""
    # Test valid hook file
    assert valid_hook('pre_prompt.py', 'pre_prompt') is True
    assert valid_hook('pre_gen_project.sh', 'pre_gen_project') is True
    assert valid_hook('post_gen_project.bat', 'post_gen_project') is True
    
    # Test valid hook with different extensions
    assert valid_hook('pre_prompt', 'pre_prompt') is True
    assert valid_hook('pre_gen_project', 'pre_gen_project') is True
    
    # Test hook name mismatch
    assert valid_hook('pre_prompt.py', 'post_gen_project') is False
    assert valid_hook('pre_gen_project.sh', 'pre_prompt') is False
    
    # Test unsupported hook name
    assert valid_hook('unsupported_hook.py', 'unsupported_hook') is False
    assert valid_hook('invalid_hook.sh', 'invalid_hook') is False
    
    # Test backup files (should be invalid)
    assert valid_hook('pre_prompt.py~', 'pre_prompt') is False
    assert valid_hook('pre_gen_project.sh~', 'pre_gen_project') is False
    assert valid_hook('post_gen_project~', 'post_gen_project') is False
    
    # Test files with path separators
    assert valid_hook('/path/to/pre_prompt.py', 'pre_prompt') is True
    assert valid_hook('hooks/pre_gen_project.sh', 'pre_gen_project') is True
    
    # Test case sensitivity
    assert valid_hook('Pre_Prompt.py', 'pre_prompt') is False
    assert valid_hook('PRE_PROMPT.py', 'pre_prompt') is False
    
    # Test empty and special filenames
    assert valid_hook('.pre_prompt', 'pre_prompt') is False
    assert valid_hook('pre_prompt.', 'pre_prompt') is True
    
    # Test multiple dots in filename
    assert valid_hook('pre_prompt.backup.py', 'pre_prompt') is False
    assert valid_hook('pre_prompt.test.sh', 'pre_prompt') is False


# LLM-generated content at query #13
#--------------------------

```python
def test_find_hook(tmp_path):
    """Test find_hook function with various scenarios."""
    # Create a hooks directory
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Test 1: No hooks directory exists
    with work_in(tmp_path):
        result = find_hook('pre_prompt', 'nonexistent_hooks')
        assert result is None
    
    # Test 2: Empty hooks directory
    with work_in(tmp_path):
        result = find_hook('pre_prompt', str(hooks_dir))
        assert result is None
    
    # Test 3: Hook file with matching name and supported hook type
    hook_file = hooks_dir / "pre_prompt.sh"
    hook_file.write_text("#!/bin/bash\necho 'test'")
    
    with work_in(tmp_path):
        result = find_hook('pre_prompt', str(hooks_dir))
        assert result is not None
        assert len(result) == 1
        assert str(hook_file) in result[0]
    
    # Test 4: Multiple hook files with same name but different extensions
    hook_file_py = hooks_dir / "pre_prompt.py"
    hook_file_py.write_text("print('test')")
    
    with work_in(tmp_path):
        result = find_hook('pre_prompt', str(hooks_dir))
        assert result is not None
        assert len(result) == 2
    
    # Test 5: Backup files should be ignored
    backup_file = hooks_dir / "pre_prompt.sh~"
    backup_file.write_text("#!/bin/bash\necho 'backup'")
    
    with work_in(tmp_path):
        result = find_hook('pre_prompt', str(hooks_dir))
        assert result is not None
        assert len(result) == 2  # Still 2, backup file ignored
    
    # Test 6: Non-matching hook name
    with work_in(tmp_path):
        result = find_hook('post_gen_project', str(hooks_dir))
        assert result is None
    
    # Test 7: Unsupported hook type (not in _HOOKS)
    unsupported_hook = hooks_dir / "unsupported_hook.sh"
    unsupported_hook.write_text("#!/bin/bash\necho 'test'")
    
    with work_in(tmp_path):
        result = find_hook('unsupported_hook', str(hooks_dir))
        assert result is None
    
    # Test 8: File without extension matching hook name
    hook_no_ext = hooks_dir / "post_gen_project"
    hook_no_ext.write_text("#!/bin/bash\necho 'test'")
    
    with work_in(tmp_path):
        result = find_hook('post_gen_project', str(hooks_dir))
        assert result is not None
        assert len(result) == 1
        assert str(hook_no_ext) in result[0]


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hooks and cleans up on failure."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, False)
    mock_run_hook.assert_called_once_with('pre_gen_project', project_dir, context)
    
    # Test failed hook with delete_project_on_failure=True
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    mock_logger = mocker.patch('cookiecutter.hooks.logger')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, True)
    
    mock_rmtree.assert_called_once_with(project_dir)
    mock_logger.exception.assert_called_once()
    
    # Test failed hook with delete_project_on_failure=False
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_logger.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Undefined variable')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, False)
    
    mock_rmtree.assert_not_called()
    mock_logger.exception.assert_called_once()


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and handles failures."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False,
    )
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test FailedHookException with delete_project_on_failure=True
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test FailedHookException with delete_project_on_failure=False
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False,
        )
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError with delete_project_on_failure=True
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True,
        )
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script function executes scripts correctly."""
    import sys
    import stat
    
    # Test successful script execution
    script_file = tmp_path / "test_script.py"
    script_file.write_text("exit(0)")
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))
    
    # Test script with non-zero exit status
    script_file_fail = tmp_path / "test_script_fail.py"
    script_file_fail.write_text("exit(1)")
    
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(script_file_fail), cwd=str(tmp_path))
    
    # Test shell script (bash/batch)
    if sys.platform.startswith('win'):
        shell_script = tmp_path / "test_script.bat"
        shell_script.write_text("@echo off\nexit /b 0")
    else:
        shell_script = tmp_path / "test_script.sh"
        shell_script.write_text("#!/bin/bash\nexit 0")
        os.chmod(shell_script, os.stat(shell_script).st_mode | stat.S_IEXEC)
    
    run_script(str(shell_script), cwd=str(tmp_path))
    
    # Test ENOEXEC error (empty file without shebang on Unix)
    if not sys.platform.startswith('win'):
        empty_script = tmp_path / "empty_script.sh"
        empty_script.write_text("")
        os.chmod(empty_script, os.stat(empty_script).st_mode | stat.S_IEXEC)
        
        with pytest.raises(FailedHookException, match="might be an empty file or missing a shebang"):
            run_script(str(empty_script), cwd=str(tmp_path))
    
    # Test with default cwd
    script_default_cwd = tmp_path / "test_script_default.py"
    script_default_cwd.write_text("exit(0)")
    run_script(str(script_default_cwd))
    
    # Test OSError handling
    nonexistent_script = tmp_path / "nonexistent.py"
    with pytest.raises(FailedHookException, match="Hook script failed"):
        run_script(str(nonexistent_script), cwd=str(tmp_path))


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, mocker):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found and executes successfully
    repo_dir2 = tmp_path / "test_repo2"
    repo_dir2.mkdir()
    hooks_dir2 = repo_dir2 / "hooks"
    hooks_dir2.mkdir()
    
    script_path = hooks_dir2 / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    script_path.chmod(0o755)
    
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir2)
    mocker.patch('cookiecutter.hooks.run_script')
    
    result = run_pre_prompt_hook(repo_dir2)
    assert result == repo_dir2
    
    # Test 3: pre_prompt hook fails with FailedHookException
    repo_dir3 = tmp_path / "test_repo3"
    repo_dir3.mkdir()
    hooks_dir3 = repo_dir3 / "hooks"
    hooks_dir3.mkdir()
    
    script_path3 = hooks_dir3 / "pre_prompt.sh"
    script_path3.write_text("#!/bin/bash\nexit 1")
    script_path3.chmod(0o755)
    
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir3)
    mocker.patch('cookiecutter.hooks.run_script', side_effect=FailedHookException('Hook failed'))
    
    with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
        run_pre_prompt_hook(repo_dir3)
    
    # Test 4: pre_prompt hook as Python script
    repo_dir4 = tmp_path / "test_repo4"
    repo_dir4.mkdir()
    hooks_dir4 = repo_dir4 / "hooks"
    hooks_dir4.mkdir()
    
    script_path4 = hooks_dir4 / "pre_prompt.py"
    script_path4.write_text("print('test')")
    
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir4)
    mocker.patch('cookiecutter.hooks.run_script')
    
    result = run_pre_prompt_hook(repo_dir4)
    assert result == repo_dir4
    
    # Test 5: Multiple pre_prompt hooks found and executed
    repo_dir5 = tmp_path / "test_repo5"
    repo_dir5.mkdir()
    hooks_dir5 = repo_dir5 / "hooks"
    hooks_dir5.mkdir()
    
    script_path5a = hooks_dir5 / "pre_prompt.sh"
    script_path5a.write_text("#!/bin/bash\necho 'test1'")
    script_path5a.chmod(0o755)
    
    script_path5b = hooks_dir5 / "pre_prompt.py"
    script_path5b.write_text("print('test2')")
    
    mocker.patch('cookiecutter.hooks.create_tmp_repo_dir', return_value=repo_dir5)
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    result = run_pre_prompt_hook(repo_dir5)
    assert result == repo_dir5
    assert mock_run_script.call_count == 2


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from cookiecutter.exceptions import FailedHookException


def test_run_hook():
    """Test run_hook function executes hook scripts found in hooks directory."""
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mock_script_path = '/path/to/hooks/pre_gen_project.sh'
    
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        
        mock_find_hook.return_value = [mock_script_path]
        
        run_hook(hook_name, project_dir, context)
        
        mock_find_hook.assert_called_once_with(hook_name)
        mock_run_script.assert_called_once_with(mock_script_path, project_dir, context)


def test_run_hook_no_scripts_found():
    """Test run_hook when no hook scripts are found."""
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        
        mock_find_hook.return_value = None
        
        run_hook(hook_name, project_dir, context)
        
        mock_find_hook.assert_called_once_with(hook_name)
        mock_run_script.assert_not_called()


def test_run_hook_multiple_scripts():
    """Test run_hook executes multiple hook scripts."""
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    mock_scripts = [
        '/path/to/hooks/pre_gen_project.sh',
        '/path/to/hooks/pre_gen_project.py'
    ]
    
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        
        mock_find_hook.return_value = mock_scripts
        
        run_hook(hook_name, project_dir, context)
        
        mock_find_hook.assert_called_once_with(hook_name)
        assert mock_run_script.call_count == 2
        mock_run_script.assert_any_call(mock_scripts[0], project_dir, context)
        mock_run_script.assert_any_call(mock_scripts[1], project_dir, context)


def test_run_hook_empty_scripts_list():
    """Test run_hook when find_hook returns empty list."""
    hook_name = 'pre_prompt'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook, \
         patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
        
        mock_find_hook.return_value = []
        
        run_hook(hook_name, project_dir, context)
        
        mock_find_hook.assert_called_once_with(hook_name)
        mock_run_script.assert_not_called()


# LLM-generated content at query #19
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a test script with Jinja2 template syntax
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    # Mock run_script to verify it's called with the rendered script
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    # Define context
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Call the function
    run_script_with_context(script_path, tmp_path, context)
    
    # Verify run_script was called with a temporary file path and cwd
    assert mock_run_script.called
    temp_script_path = mock_run_script.call_args[0][0]
    assert mock_run_script.call_args[0][1] == tmp_path
    
    # Verify the temporary file contains rendered content
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'my_project' in rendered_content
    assert '{{ cookiecutter.project_name }}' not in rendered_content


def test_run_script_with_context_python_file(tmp_path, mocker):
    """Test run_script_with_context with a Python script file."""
    script_content = 'print("{{ variable }}")'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'variable': 'test_value'}
    
    run_script_with_context(script_path, tmp_path, context)
    
    assert mock_run_script.called
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.py')
    
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'test_value' in rendered_content


def test_run_script_with_context_with_undefined_variable(tmp_path, mocker):
    """Test run_script_with_context handles undefined variables gracefully."""
    script_content = '#!/bin/bash\necho "{{ undefined_var }}"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'other_var': 'value'}
    
    # Should raise UndefinedError when rendering undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, tmp_path, context)


def test_run_script_with_context_cwd_parameter(tmp_path, mocker):
    """Test run_script_with_context passes correct cwd to run_script."""
    script_content = 'echo "test"'
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    custom_cwd = tmp_path / "custom_dir"
    context = {}
    
    run_script_with_context(script_path, custom_cwd, context)
    
    assert mock_run_script.call_args[0][1] == custom_cwd


def test_run_script_with_context_complex_template(tmp_path, mocker):
    """Test run_script_with_context with complex Jinja2 template."""
    script_content = '''#!/bin/bash
{% for item in items %}
echo "{{ item }}"
{% endfor %}
echo "{{ name }}"
'''
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'items': ['a', 'b', 'c'], 'name': 'test'}
    
    run_script_with_context(script_path, tmp_path, context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    
    assert 'echo "a"' in rendered_content
    assert 'echo "b"' in rendered_content
    assert 'echo "c"' in rendered_content
    assert 'echo "test"' in rendered_content
    assert '{% for' not in rendered_content


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script(tmp_path, monkeypatch):
    """Test run_script executes scripts correctly."""
    import stat
    
    # Test successful script execution
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)
    
    # Should not raise any exception
    run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_python():
    """Test run_script executes Python scripts correctly."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import sys\nsys.exit(0)")
        f.flush()
        
        try:
            run_script(f.name)
        finally:
            os.unlink(f.name)


def test_run_script_non_zero_exit(tmp_path):
    """Test run_script raises exception on non-zero exit status."""
    import stat
    
    script_file = tmp_path / "failing_script.sh"
    script_file.write_text("#!/bin/bash\nexit 1")
    script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_missing_shebang(tmp_path):
    """Test run_script raises exception for script with missing shebang."""
    import stat
    
    script_file = tmp_path / "no_shebang.sh"
    script_file.write_text("echo hello")
    script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)
    
    # On some systems, ENOEXEC is raised for missing shebang
    with pytest.raises(FailedHookException):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_with_cwd(tmp_path):
    """Test run_script executes script in specified working directory."""
    import stat
    
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\ntouch output.txt\nexit 0")
    script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)
    
    run_script(str(script_file), cwd=str(work_dir))
    
    # Verify the script was executed in the correct directory
    assert (work_dir / "output.txt").exists()


def test_run_script_python_with_context(tmp_path):
    """Test run_script executes Python scripts with proper exit handling."""
    python_script = tmp_path / "test_script.py"
    python_script.write_text("print('Hello')\nexit(0)")
    
    run_script(str(python_script), cwd=str(tmp_path))


def test_run_script_oserror(tmp_path, monkeypatch):
    """Test run_script handles OSError exceptions."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Exec format error")
    
    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    
    with pytest.raises(FailedHookException, match='might be an empty file or missing a shebang'):
        run_script(str(script_file), cwd=str(tmp_path))


def test_run_script_generic_oserror(tmp_path, monkeypatch):
    """Test run_script handles generic OSError exceptions."""
    script_file = tmp_path / "test_script.sh"
    script_file.write_text("#!/bin/bash\nexit 0")
    
    def mock_popen(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")
    
    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    
    with pytest.raises(FailedHookException, match='Hook script failed'):
        run_script(str(script_file), cwd=str(tmp_path))


# LLM-generated content at query #21
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja2 template syntax
    script_content = '#!/bin/bash\necho "{{ cookiecutter.project_name }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)
    
    # Create context dictionary
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    # Mock run_script to verify it was called
    mock_run_script_called = []
    original_run_script = None
    
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append((script_path, cwd))
        # Verify the temporary file exists and has rendered content
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert 'test_project' in temp_content
        assert '{{ cookiecutter.project_name }}' not in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called
    assert len(mock_run_script_called) == 1
    script_path, cwd = mock_run_script_called[0]
    assert str(tmp_path) == cwd


def test_run_script_with_context_python_file(tmp_path, monkeypatch):
    """Test run_script_with_context with a Python script."""
    script_content = '#!/usr/bin/env python\nprint("{{ cookiecutter.name }}")'
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'name': 'myproject'}}
    
    mock_run_script_called = []
    
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append((script_path, cwd))
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert 'myproject' in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(mock_run_script_called) == 1


def test_run_script_with_context_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context with complex Jinja2 template."""
    script_content = '''#!/bin/bash
{% if cookiecutter.use_docker %}
docker build -t {{ cookiecutter.project_name }} .
{% endif %}
echo "Project: {{ cookiecutter.project_name }}"
'''
    script_file = tmp_path / "complex_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)
    
    context = {
        'cookiecutter': {
            'project_name': 'my_app',
            'use_docker': True
        }
    }
    
    mock_run_script_called = []
    
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append(script_path)
        temp_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_app' in temp_content
        assert 'docker build' in temp_content
        assert '{{' not in temp_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(mock_run_script_called) == 1


def test_run_script_with_context_pathlib_path(tmp_path, monkeypatch):
    """Test run_script_with_context accepts Path objects."""
    script_content = '#!/bin/bash\necho "{{ cookiecutter.value }}"'
    script_file = tmp_path / "test_script.sh"
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)
    
    context = {'cookiecutter': {'value': 'test_value'}}
    
    mock_run_script_called = []
    
    def mock_run_script(script_path, cwd):
        mock_run_script_called.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    # Use Path objects instead of strings
    run_script_with_context(script_file, tmp_path, context)
    
    assert len(mock_run_script_called) == 1


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves file extension."""
    script_content = '#!/usr/bin/env python\nprint("test")'
    script_file = tmp_path / "test_script.py"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {}}
    
    temp_file_extensions = []
    
    def mock_run_script(script_path, cwd):
        _, ext = os.path.splitext(script_path)
        temp_file_extensions.append(ext)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    assert len(temp_file_extensions) == 1
    assert temp_file_extensions[0] == '.py'


# LLM-generated content at query #22
#--------------------------

```python
def test_run_hook(tmp_path, monkeypatch):
    """Test run_hook function executes hook scripts correctly."""
    # Setup
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create a simple Python hook script
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("# Test hook\nprint('Hook executed')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'project_name': 'test_project',
        'author': 'Test Author'
    }
    
    # Change to hooks directory so find_hook works
    monkeypatch.chdir(tmp_path)
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Verify - no exception should be raised


def test_run_hook_no_scripts_found(tmp_path, monkeypatch):
    """Test run_hook when no hook scripts are found."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    # Should return without error when no scripts found
    run_hook('pre_gen_project', project_dir, context)


def test_run_hook_with_context_rendering(tmp_path, monkeypatch):
    """Test run_hook renders context variables in hook scripts."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create hook with Jinja template
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("# Project: {{ project_name }}\nprint('{{ project_name }}')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {
        'project_name': 'my_project',
        'author': 'Test'
    }
    
    monkeypatch.chdir(tmp_path)
    
    run_hook('pre_gen_project', project_dir, context)


def test_run_hook_failed_execution(tmp_path, monkeypatch):
    """Test run_hook raises FailedHookException on script failure."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create hook script that fails
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("import sys\nsys.exit(1)")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(FailedHookException):
        run_hook('pre_gen_project', project_dir, context)


def test_run_hook_multiple_scripts(tmp_path, monkeypatch):
    """Test run_hook executes multiple hook scripts in sequence."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    # Create multiple hook scripts with same name pattern
    hook_script1 = hooks_dir / "pre_gen_project.py"
    hook_script1.write_text("print('Hook 1')")
    
    hook_script2 = hooks_dir / "pre_gen_project.sh"
    hook_script2.write_text("#!/bin/bash\necho 'Hook 2'")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {'project_name': 'test_project'}
    
    monkeypatch.chdir(tmp_path)
    
    run_hook('pre_gen_project', project_dir, context)


# LLM-generated content at query #23
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return original repo_dir
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found, should create temp dir and run script
    repo_dir = tmp_path / "test_repo_with_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a valid pre_prompt script
    script_path = hooks_dir / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    script_path.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert os.path.isdir(result)
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir = tmp_path / "test_repo_failing_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a failing pre_prompt script
    script_path = hooks_dir / "pre_prompt.sh"
    script_path.write_text("#!/bin/bash\nexit 1")
    script_path.chmod(0o755)
    
    with pytest.raises(FailedHookException, match='Pre-Prompt Hook script failed'):
        run_pre_prompt_hook(repo_dir)
    
    # Test 4: pre_prompt Python script
    repo_dir = tmp_path / "test_repo_python_hook"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create a valid pre_prompt Python script
    script_path = hooks_dir / "pre_prompt.py"
    script_path.write_text("print('test')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert os.path.isdir(result)
    
    # Test 5: Multiple pre_prompt hooks
    repo_dir = tmp_path / "test_repo_multiple_hooks"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    # Create multiple pre_prompt scripts
    script1 = hooks_dir / "pre_prompt.sh"
    script1.write_text("#!/bin/bash\necho 'test1'")
    script1.chmod(0o755)
    
    script2 = hooks_dir / "pre_prompt.py"
    script2.write_text("print('test2')")
    
    result = run_pre_prompt_hook(repo_dir)
    assert result != repo_dir
    assert os.path.isdir(result)


# LLM-generated content at query #24
#--------------------------

```python
def test_run_hook_from_repo_dir(tmp_path, mocker):
    """Test run_hook_from_repo_dir executes hook and cleans up on failure."""
    repo_dir = tmp_path / "repo"
    project_dir = tmp_path / "project"
    repo_dir.mkdir()
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Test successful hook execution
    mock_run_hook = mocker.patch('cookiecutter.hooks.run_hook')
    run_hook_from_repo_dir(
        repo_dir=repo_dir,
        hook_name='post_gen_project',
        project_dir=project_dir,
        context=context,
        delete_project_on_failure=False
    )
    mock_run_hook.assert_called_once_with('post_gen_project', project_dir, context)
    
    # Test hook failure with project deletion
    mock_run_hook.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    mock_rmtree = mocker.patch('cookiecutter.hooks.rmtree')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    mock_rmtree.assert_called_once_with(project_dir)
    
    # Test hook failure without project deletion
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = FailedHookException('Hook failed')
    
    with pytest.raises(FailedHookException):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=False
        )
    mock_rmtree.assert_not_called()
    
    # Test UndefinedError handling
    mock_run_hook.reset_mock()
    mock_rmtree.reset_mock()
    mock_run_hook.side_effect = UndefinedError('Variable undefined')
    
    with pytest.raises(UndefinedError):
        run_hook_from_repo_dir(
            repo_dir=repo_dir,
            hook_name='post_gen_project',
            project_dir=project_dir,
            context=context,
            delete_project_on_failure=True
        )
    mock_rmtree.assert_called_once_with(project_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context executes a script with Jinja rendering."""
    # Create a temporary script with Jinja template syntax
    script_content = "#!/usr/bin/env python\nprint('{{ project_name }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'my_project'}
    cwd = tmp_path
    
    # Mock run_script to verify it's called with rendered content
    called_with = []
    original_run_script = run_script
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append((script_path, cwd))
        # Verify the rendered content
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_project' in rendered_content
        assert '{{' not in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(script_path, cwd, context)
    
    assert len(called_with) == 1
    assert called_with[0][1] == cwd


def test_run_script_with_context_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context raises UndefinedError for missing context."""
    script_content = "#!/usr/bin/env python\nprint('{{ missing_var }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'my_project'}
    cwd = tmp_path
    
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, cwd, context)


def test_run_script_with_context_bash_script(tmp_path, monkeypatch):
    """Test run_script_with_context with bash script extension."""
    script_content = "#!/bin/bash\necho '{{ project_name }}'\n"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'project_name': 'bash_project'}
    cwd = tmp_path
    
    called_with = []
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append((script_path, cwd))
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'bash_project' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(script_path, cwd, context)
    
    assert len(called_with) == 1
    assert called_with[0][0].endswith('.sh')


def test_run_script_with_context_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple context variables."""
    script_content = "#!/usr/bin/env python\nprint('{{ name }}-{{ version }}')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'name': 'myapp', 'version': '1.0.0'}
    cwd = tmp_path
    
    called_with = []
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append(script_path)
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'myapp-1.0.0' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(script_path, cwd, context)
    
    assert len(called_with) == 1


def test_run_script_with_context_empty_context(tmp_path, monkeypatch):
    """Test run_script_with_context with empty context."""
    script_content = "#!/usr/bin/env python\nprint('hello')\n"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    cwd = tmp_path
    
    called_with = []
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(script_path, cwd, context)
    
    assert len(called_with) == 1


def test_run_script_with_context_complex_jinja(tmp_path, monkeypatch):
    """Test run_script_with_context with complex Jinja logic."""
    script_content = """#!/usr/bin/env python
{% if debug %}
print('Debug mode')
{% else %}
print('Production mode')
{% endif %}
"""
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'debug': True}
    cwd = tmp_path
    
    called_with = []
    
    def mock_run_script(script_path, cwd='.'):
        called_with.append(script_path)
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'Debug mode' in rendered_content
        assert 'Production mode' not in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    run_script_with_context(script_path, cwd, context)
    
    assert len(called_with) == 1


# LLM-generated content at query #26
#--------------------------

```python
def test_run_pre_prompt_hook(tmp_path, monkeypatch):
    """Test run_pre_prompt_hook function."""
    # Test 1: No pre_prompt hook found, should return repo_dir unchanged
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir
    
    # Test 2: pre_prompt hook found and executes successfully
    repo_dir_2 = tmp_path / "template2"
    repo_dir_2.mkdir()
    hooks_dir_2 = repo_dir_2 / "hooks"
    hooks_dir_2.mkdir()
    
    hook_script = hooks_dir_2 / "pre_prompt.sh"
    hook_script.write_text("#!/bin/bash\nexit 0")
    hook_script.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir_2)
    assert result != repo_dir_2  # Should return a temporary directory
    assert isinstance(result, (str, Path))
    
    # Test 3: pre_prompt hook fails, should raise FailedHookException
    repo_dir_3 = tmp_path / "template3"
    repo_dir_3.mkdir()
    hooks_dir_3 = repo_dir_3 / "hooks"
    hooks_dir_3.mkdir()
    
    hook_script_3 = hooks_dir_3 / "pre_prompt.sh"
    hook_script_3.write_text("#!/bin/bash\nexit 1")
    hook_script_3.chmod(0o755)
    
    with pytest.raises(FailedHookException, match="Pre-Prompt Hook script failed"):
        run_pre_prompt_hook(repo_dir_3)
    
    # Test 4: pre_prompt Python hook found and executes successfully
    repo_dir_4 = tmp_path / "template4"
    repo_dir_4.mkdir()
    hooks_dir_4 = repo_dir_4 / "hooks"
    hooks_dir_4.mkdir()
    
    hook_script_4 = hooks_dir_4 / "pre_prompt.py"
    hook_script_4.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    hook_script_4.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir_4)
    assert result != repo_dir_4
    assert isinstance(result, (str, Path))
    
    # Test 5: Multiple pre_prompt hooks, all execute successfully
    repo_dir_5 = tmp_path / "template5"
    repo_dir_5.mkdir()
    hooks_dir_5 = repo_dir_5 / "hooks"
    hooks_dir_5.mkdir()
    
    hook_script_5a = hooks_dir_5 / "pre_prompt.sh"
    hook_script_5a.write_text("#!/bin/bash\nexit 0")
    hook_script_5a.chmod(0o755)
    
    hook_script_5b = hooks_dir_5 / "pre_prompt.py"
    hook_script_5b.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)")
    hook_script_5b.chmod(0o755)
    
    result = run_pre_prompt_hook(repo_dir_5)
    assert result != repo_dir_5
    assert isinstance(result, (str, Path))


# LLM-generated content at query #27
#--------------------------

```python
def test_run_hook(mocker, tmp_path):
    """Test run_hook executes hook scripts found in the hooks directory."""
    # Setup
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("print('Hook executed')")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Mock find_hook to return our test script
    mock_find_hook = mocker.patch(
        'cookiecutter.hooks.find_hook',
        return_value=[str(hook_script)]
    )
    
    # Mock run_script_with_context to verify it's called
    mock_run_script = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Assert
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_run_script.assert_called_once_with(
        str(hook_script),
        project_dir,
        context
    )


def test_run_hook_no_scripts_found(mocker, tmp_path):
    """Test run_hook returns early when no hook scripts are found."""
    project_dir = tmp_path / "project"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Mock find_hook to return None
    mock_find_hook = mocker.patch(
        'cookiecutter.hooks.find_hook',
        return_value=None
    )
    
    # Mock run_script_with_context to verify it's not called
    mock_run_script = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Assert
    mock_find_hook.assert_called_once_with('pre_gen_project')
    mock_run_script.assert_not_called()


def test_run_hook_multiple_scripts(mocker, tmp_path):
    """Test run_hook executes multiple hook scripts."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    
    hook_script1 = hooks_dir / "pre_gen_project.py"
    hook_script1.write_text("print('Hook 1')")
    
    hook_script2 = hooks_dir / "pre_gen_project.sh"
    hook_script2.write_text("#!/bin/bash\necho 'Hook 2'")
    
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Mock find_hook to return both scripts
    mock_find_hook = mocker.patch(
        'cookiecutter.hooks.find_hook',
        return_value=[str(hook_script1), str(hook_script2)]
    )
    
    # Mock run_script_with_context to verify it's called for each script
    mock_run_script = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook('pre_gen_project', project_dir, context)
    
    # Assert
    assert mock_run_script.call_count == 2
    mock_run_script.assert_any_call(str(hook_script1), project_dir, context)
    mock_run_script.assert_any_call(str(hook_script2), project_dir, context)


# LLM-generated content at query #28
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context executes a script with rendered context."""
    # Create a temporary script file with Jinja template
    script_path = tmp_path / "test_script.py"
    script_content = """
import os
with open('{{ output_file }}', 'w') as f:
    f.write('{{ greeting }} {{ name }}')
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    # Define context with variables to be rendered
    context = {
        'output_file': 'output.txt',
        'greeting': 'Hello',
        'name': 'World'
    }
    
    # Run the script with context
    run_script_with_context(script_path, tmp_path, context)
    
    # Verify the script was executed and created the output file
    output_file = tmp_path / 'output.txt'
    assert output_file.exists()
    assert output_file.read_text() == 'Hello World'


def test_run_script_with_context_shell_script(tmp_path, monkeypatch):
    """Test run_script_with_context with a shell script."""
    # Create a temporary shell script with Jinja template
    script_path = tmp_path / "test_script.sh"
    script_content = """#!/bin/bash
echo "{{ message }}" > {{ output_file }}
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'message': 'Test message',
        'output_file': 'shell_output.txt'
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    output_file = tmp_path / 'shell_output.txt'
    assert output_file.exists()
    assert 'Test message' in output_file.read_text()


def test_run_script_with_context_complex_jinja(tmp_path):
    """Test run_script_with_context with complex Jinja template."""
    script_path = tmp_path / "test_script.py"
    script_content = """
result = '{% if condition %}yes{% else %}no{% endif %}'
with open('{{ filename }}', 'w') as f:
    f.write(result)
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'condition': True,
        'filename': 'result.txt'
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    result_file = tmp_path / 'result.txt'
    assert result_file.exists()
    assert result_file.read_text() == 'yes'


def test_run_script_with_context_preserves_extension(tmp_path):
    """Test that run_script_with_context preserves file extension."""
    script_path = tmp_path / "test_script.py"
    script_content = """
with open('{{ output }}', 'w') as f:
    f.write('done')
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {'output': 'done.txt'}
    
    run_script_with_context(script_path, tmp_path, context)
    
    output_file = tmp_path / 'done.txt'
    assert output_file.exists()
    assert output_file.read_text() == 'done'


def test_run_script_with_context_with_special_chars(tmp_path):
    """Test run_script_with_context with special characters in context."""
    script_path = tmp_path / "test_script.py"
    script_content = """
with open('{{ filename }}', 'w') as f:
    f.write('{{ text }}')
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {
        'filename': 'special.txt',
        'text': 'Special chars: !@#$%^&*()'
    }
    
    run_script_with_context(script_path, tmp_path, context)
    
    output_file = tmp_path / 'special.txt'
    assert output_file.read_text() == 'Special chars: !@#$%^&*()'


def test_run_script_with_context_different_cwd(tmp_path):
    """Test run_script_with_context runs from specified working directory."""
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    
    script_path = script_dir / "test_script.py"
    script_content = """
import os
with open('output.txt', 'w') as f:
    f.write(os.getcwd())
"""
    script_path.write_text(script_content, encoding='utf-8')
    
    context = {}
    run_script_with_context(script_path, work_dir, context)
    
    output_file = work_dir / 'output.txt'
    assert output_file.exists()
    assert str(work_dir) in output_file.read_text()


# LLM-generated content at query #29
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: No hooks directory exists
    result = find_hook('pre_prompt', 'hooks')
    assert result is None
    
    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None
    
    # Test 3: Hook file exists with matching name
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.resolve())
    
    # Test 4: Multiple hook files with same name but different extensions
    hook_file_py = hooks_dir / 'pre_prompt.py'
    hook_file_py.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    
    # Test 5: Backup file should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.resolve()) not in result
    
    # Test 6: Non-matching hook name
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is None
    
    # Test 7: Invalid hook name (not in _HOOKS)
    invalid_hook = hooks_dir / 'invalid_hook.sh'
    invalid_hook.write_text('#!/bin/bash\necho "invalid"')
    result = find_hook('invalid_hook', str(hooks_dir))
    assert result is None
    
    # Test 8: Valid hook with different extension
    post_hook = hooks_dir / 'post_gen_project.py'
    post_hook.write_text('print("post")')
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert str(post_hook.resolve()) in result


# LLM-generated content at query #30
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context executes script after Jinja rendering."""
    # Create a temporary script file with Jinja template
    script_file = tmp_path / "test_script.py"
    script_content = "# Context: {{ cookiecutter.project_name }}\nprint('{{ cookiecutter.project_name }}')"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Mock run_script to verify it's called
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    # Call the function
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify run_script was called once
    assert mock_run_script.call_count == 1
    
    # Get the temporary file path that was passed to run_script
    temp_script_path = mock_run_script.call_args[0][0]
    
    # Verify the temporary file contains rendered content
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    assert 'my_project' in rendered_content
    assert '{{ cookiecutter.project_name }}' not in rendered_content
    
    # Clean up temporary file
    Path(temp_script_path).unlink()


def test_run_script_with_context_with_undefined_variable(tmp_path, mocker):
    """Test run_script_with_context handles undefined Jinja variables."""
    script_file = tmp_path / "test_script.py"
    script_content = "# {{ undefined_var }}"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Should raise UndefinedError when rendering undefined variables
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_file), str(tmp_path), context)


def test_run_script_with_context_preserves_extension(tmp_path, mocker):
    """Test run_script_with_context preserves file extension."""
    script_file = tmp_path / "test_script.sh"
    script_content = "#!/bin/bash\necho '{{ cookiecutter.name }}'"
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {'cookiecutter': {'name': 'test'}}
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    # Verify the temporary file has .sh extension
    temp_script_path = mock_run_script.call_args[0][0]
    assert temp_script_path.endswith('.sh')
    
    Path(temp_script_path).unlink()


def test_run_script_with_context_multiple_variables(tmp_path, mocker):
    """Test run_script_with_context with multiple context variables."""
    script_file = tmp_path / "test_script.py"
    script_content = """
# Project: {{ project_name }}
# Author: {{ author }}
# Version: {{ version }}
"""
    script_file.write_text(script_content, encoding='utf-8')
    
    context = {
        'project_name': 'MyProject',
        'author': 'John Doe',
        'version': '1.0.0'
    }
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_file), str(tmp_path), context)
    
    temp_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(temp_script_path).read_text(encoding='utf-8')
    
    assert 'MyProject' in rendered_content
    assert 'John Doe' in rendered_content
    assert '1.0.0' in rendered_content
    
    Path(temp_script_path).unlink()


def test_run_script_with_context_cwd_parameter(tmp_path, mocker):
    """Test run_script_with_context passes correct cwd to run_script."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("print('test')", encoding='utf-8')
    
    custom_cwd = tmp_path / "custom_dir"
    custom_cwd.mkdir()
    context = {}
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    run_script_with_context(str(script_file), str(custom_cwd), context)
    
    # Verify cwd was passed correctly
    assert mock_run_script.call_args[0][1] == str(custom_cwd)
    
    temp_script_path = mock_run_script.call_args[0][0]
    Path(temp_script_path).unlink()


# LLM-generated content at query #31
#--------------------------

```python
def test_run_hook(tmp_path, mocker):
    """Test run_hook function."""
    # Setup
    hook_name = 'pre_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project', 'author': 'test_author'}
    
    # Mock find_hook to return a script
    mock_script = tmp_path / 'hook_script.py'
    mock_script.write_text('print("hook executed")')
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = [str(mock_script)]
    
    # Mock run_script_with_context
    mock_run_script_with_context = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook(hook_name, project_dir, context)
    
    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    mock_run_script_with_context.assert_called_once_with(
        str(mock_script), project_dir, context
    )


def test_run_hook_no_scripts_found(mocker):
    """Test run_hook when no scripts are found."""
    hook_name = 'pre_gen_project'
    project_dir = '/some/path'
    context = {}
    
    # Mock find_hook to return None
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = None
    
    # Mock run_script_with_context should not be called
    mock_run_script_with_context = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook(hook_name, project_dir, context)
    
    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    mock_run_script_with_context.assert_not_called()


def test_run_hook_multiple_scripts(tmp_path, mocker):
    """Test run_hook with multiple hook scripts."""
    hook_name = 'post_gen_project'
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    context = {'project_name': 'test_project'}
    
    # Mock find_hook to return multiple scripts
    mock_scripts = [str(tmp_path / 'script1.py'), str(tmp_path / 'script2.sh')]
    
    mock_find_hook = mocker.patch('cookiecutter.hooks.find_hook')
    mock_find_hook.return_value = mock_scripts
    
    # Mock run_script_with_context
    mock_run_script_with_context = mocker.patch(
        'cookiecutter.hooks.run_script_with_context'
    )
    
    # Execute
    run_hook(hook_name, project_dir, context)
    
    # Assert
    mock_find_hook.assert_called_once_with(hook_name)
    assert mock_run_script_with_context.call_count == 2
    mock_run_script_with_context.assert_any_call(
        mock_scripts[0], project_dir, context
    )
    mock_run_script_with_context.assert_any_call(
        mock_scripts[1], project_dir, context
    )


# LLM-generated content at query #32
#--------------------------

```python
def test_run_script_with_context(tmp_path, mocker):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script with Jinja template syntax
    script_path = tmp_path / "test_script.py"
    script_content = "print('{{ cookiecutter.project_name }}')\n"
    script_path.write_text(script_content, encoding='utf-8')
    
    # Mock run_script to verify it gets called with correct path
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    # Verify run_script was called
    assert mock_run_script.called
    called_script_path = mock_run_script.call_args[0][0]
    called_cwd = mock_run_script.call_args[0][1]
    
    # Verify the rendered script content
    rendered_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "test_project" in rendered_content
    assert "{{ cookiecutter.project_name }}" not in rendered_content
    assert called_cwd == str(tmp_path)


def test_run_script_with_context_with_undefined_variable(tmp_path, mocker):
    """Test run_script_with_context handles undefined variables gracefully."""
    script_path = tmp_path / "test_script.py"
    script_content = "print('{{ undefined_var }}')\n"
    script_path.write_text(script_content, encoding='utf-8')
    
    mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    # Should raise UndefinedError when rendering undefined variables
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_path), str(tmp_path), context)


def test_run_script_with_context_bash_script(tmp_path, mocker):
    """Test run_script_with_context with bash script."""
    script_path = tmp_path / "test_script.sh"
    script_content = "#!/bin/bash\necho '{{ cookiecutter.name }}'\n"
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'name': 'my_project'}}
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    # Verify the temporary script was created and passed to run_script
    assert mock_run_script.called
    called_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "my_project" in rendered_content
    assert ".sh" in called_script_path


def test_run_script_with_context_complex_template(tmp_path, mocker):
    """Test run_script_with_context with complex Jinja expressions."""
    script_path = tmp_path / "test_script.py"
    script_content = (
        "{% if cookiecutter.use_feature %}\n"
        "print('Feature enabled')\n"
        "{% else %}\n"
        "print('Feature disabled')\n"
        "{% endif %}\n"
    )
    script_path.write_text(script_content, encoding='utf-8')
    
    mock_run_script = mocker.patch('cookiecutter.hooks.run_script')
    
    context = {'cookiecutter': {'use_feature': True}}
    
    run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert mock_run_script.called
    called_script_path = mock_run_script.call_args[0][0]
    rendered_content = Path(called_script_path).read_text(encoding='utf-8')
    assert "Feature enabled" in rendered_content
    assert "{% if" not in rendered_content


# LLM-generated content at query #33
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function with various scenarios."""
    # Change to temporary directory
    monkeypatch.chdir(tmp_path)
    
    # Test 1: No hooks directory exists
    result = find_hook('pre_prompt', 'hooks')
    assert result is None
    
    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is None
    
    # Test 3: Hook file with matching name and supported hook type
    hook_file = hooks_dir / 'pre_prompt.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file.absolute())
    
    # Test 4: Multiple hook files with same name but different extensions
    hook_file_py = hooks_dir / 'pre_prompt.py'
    hook_file_py.write_text('print("test")')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    
    # Test 5: Backup file should be ignored
    backup_file = hooks_dir / 'pre_prompt.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_prompt', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file.absolute()) not in result
    
    # Test 6: Non-matching hook name
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is None
    
    # Test 7: Unsupported hook type
    unsupported_file = hooks_dir / 'unsupported_hook.sh'
    unsupported_file.write_text('#!/bin/bash')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None
    
    # Test 8: Multiple different hook types
    post_gen_file = hooks_dir / 'post_gen_project.py'
    post_gen_file.write_text('print("post gen")')
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(post_gen_file.absolute())
    
    # Test 9: Non-existent hooks directory
    result = find_hook('pre_prompt', 'non_existent_hooks')
    assert result is None


# LLM-generated content at query #34
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context executes a script with rendered context."""
    # Create a temporary script with Jinja template syntax
    script_content = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    # Create a temporary working directory
    cwd = tmp_path / "work"
    cwd.mkdir()
    
    # Mock run_script to verify it's called with correct path
    called_with = {}
    original_run_script = run_script
    
    def mock_run_script(script_path, cwd='.'):
        called_with['script_path'] = script_path
        called_with['cwd'] = cwd
        # Verify the script was rendered by checking temp file exists
        assert os.path.exists(script_path)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    run_script_with_context(str(script_path), str(cwd), context)
    
    assert called_with['cwd'] == str(cwd)
    assert called_with['script_path'] is not None


def test_run_script_with_context_python_script(tmp_path, monkeypatch):
    """Test run_script_with_context with a Python script."""
    script_content = "print('{{ cookiecutter.name }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    cwd = tmp_path / "work"
    cwd.mkdir()
    
    called_args = []
    
    def mock_run_script(script_path, cwd='.'):
        called_args.append((script_path, cwd))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    context = {'cookiecutter': {'name': 'myproject'}}
    
    run_script_with_context(str(script_path), str(cwd), context)
    
    assert len(called_args) == 1
    assert called_args[0][1] == str(cwd)


def test_run_script_with_context_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context raises UndefinedError for undefined variables."""
    script_content = "echo '{{ undefined_var }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    cwd = tmp_path / "work"
    cwd.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_path), str(cwd), context)


def test_run_script_with_context_multiple_variables(tmp_path, monkeypatch):
    """Test run_script_with_context with multiple context variables."""
    script_content = "#!/bin/bash\necho '{{ cookiecutter.name }}-{{ cookiecutter.version }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    cwd = tmp_path / "work"
    cwd.mkdir()
    
    def mock_run_script(script_path, cwd='.'):
        # Verify temp file was created with rendered content
        content = Path(script_path).read_text()
        assert 'myproject' in content
        assert '1.0.0' in content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    context = {
        'cookiecutter': {
            'name': 'myproject',
            'version': '1.0.0'
        }
    }
    
    run_script_with_context(str(script_path), str(cwd), context)


def test_run_script_with_context_preserves_extension(tmp_path, monkeypatch):
    """Test run_script_with_context preserves script extension in temp file."""
    extensions_tested = []
    
    def mock_run_script(script_path, cwd='.'):
        _, ext = os.path.splitext(script_path)
        extensions_tested.append(ext)
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script)
    
    for ext in ['.sh', '.py', '.ps1']:
        script_path = tmp_path / f"test_script{ext}"
        script_path.write_text("echo 'test'")
        
        context = {'cookiecutter': {'name': 'test'}}
        run_script_with_context(str(script_path), str(tmp_path), context)
    
    assert extensions_tested == ['.sh', '.py', '.ps1']


# LLM-generated content at query #35
#--------------------------

```python
def test_find_hook(tmp_path, monkeypatch):
    """Test find_hook function."""
    # Test 1: No hooks directory exists
    monkeypatch.chdir(tmp_path)
    result = find_hook('pre_gen_project')
    assert result is None

    # Test 2: Hooks directory exists but is empty
    hooks_dir = tmp_path / 'hooks'
    hooks_dir.mkdir()
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is None

    # Test 3: Hook file exists and is valid
    hook_file = hooks_dir / 'pre_gen_project.sh'
    hook_file.write_text('#!/bin/bash\necho "test"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert result[0] == str(hook_file)

    # Test 4: Multiple valid hook files
    hook_file_2 = hooks_dir / 'pre_gen_project.py'
    hook_file_2.write_text('print("test")')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 2

    # Test 5: Backup file should not be included
    backup_file = hooks_dir / 'pre_gen_project.sh~'
    backup_file.write_text('#!/bin/bash\necho "backup"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 2
    assert str(backup_file) not in result

    # Test 6: Different hook name - should not find
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is None

    # Test 7: Valid post_gen_project hook
    post_hook = hooks_dir / 'post_gen_project.sh'
    post_hook.write_text('#!/bin/bash\necho "post"')
    result = find_hook('post_gen_project', str(hooks_dir))
    assert result is not None
    assert len(result) == 1
    assert str(post_hook) in result

    # Test 8: Unsupported hook name
    unsupported_hook = hooks_dir / 'unsupported_hook.sh'
    unsupported_hook.write_text('#!/bin/bash')
    result = find_hook('unsupported_hook', str(hooks_dir))
    assert result is None

    # Test 9: File without extension
    no_ext_file = hooks_dir / 'pre_gen_project'
    no_ext_file.write_text('#!/bin/bash\necho "no ext"')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert result is not None
    assert str(no_ext_file) in result

    # Test 10: Non-matching hook name in directory
    other_hook = hooks_dir / 'pre_prompt.sh'
    other_hook.write_text('#!/bin/bash')
    result = find_hook('pre_gen_project', str(hooks_dir))
    assert other_hook not in [Path(p) for p in (result or [])]


# LLM-generated content at query #36
#--------------------------

```python
def test_run_script_with_context(tmp_path, monkeypatch):
    """Test run_script_with_context renders and executes a script with context."""
    # Create a temporary script file with Jinja template syntax
    script_content = "#!/bin/bash\necho '{{ cookiecutter.project_name }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    # Create a working directory
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Define context with template variables
    context = {'cookiecutter': {'project_name': 'my_project'}}
    
    # Mock run_script to avoid actual execution
    mock_run_script = None
    executed_scripts = []
    
    def mock_run_script_impl(script_path, cwd):
        executed_scripts.append((script_path, str(cwd)))
        # Verify the temp file was created and has rendered content
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_project' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script_impl)
    
    # Execute the function
    run_script_with_context(str(script_path), cwd, context)
    
    # Verify run_script was called
    assert len(executed_scripts) == 1
    assert str(cwd) == executed_scripts[0][1]


def test_run_script_with_context_python_file(tmp_path, monkeypatch):
    """Test run_script_with_context with Python file extension."""
    script_content = "print('{{ cookiecutter.project_slug }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    context = {'cookiecutter': {'project_slug': 'my_slug'}}
    
    executed_scripts = []
    
    def mock_run_script_impl(script_path, cwd):
        executed_scripts.append(script_path)
        rendered_content = Path(script_path).read_text(encoding='utf-8')
        assert 'my_slug' in rendered_content
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script_impl)
    
    run_script_with_context(str(script_path), cwd, context)
    
    assert len(executed_scripts) == 1


def test_run_script_with_context_undefined_variable(tmp_path, monkeypatch):
    """Test run_script_with_context with undefined Jinja variable."""
    script_content = "#!/bin/bash\necho '{{ undefined_var }}'"
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    context = {'cookiecutter': {'project_name': 'test'}}
    
    # Should raise UndefinedError when trying to render undefined variable
    with pytest.raises(UndefinedError):
        run_script_with_context(str(script_path), cwd, context)


def test_run_script_with_context_complex_template(tmp_path, monkeypatch):
    """Test run_script_with_context with complex Jinja template."""
    script_content = """#!/bin/bash
{% if cookiecutter.use_feature %}
echo "Feature enabled: {{ cookiecutter.feature_name }}"
{% endif %}
"""
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding='utf-8')
    
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    context = {
        'cookiecutter': {
            'use_feature': True,
            'feature_name': 'awesome_feature'
        }
    }
    
    rendered_contents = []
    
    def mock_run_script_impl(script_path, cwd):
        rendered_contents.append(Path(script_path).read_text(encoding='utf-8'))
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script_impl)
    
    run_script_with_context(str(script_path), cwd, context)
    
    assert len(rendered_contents) == 1
    assert 'awesome_feature' in rendered_contents[0]
    assert 'Feature enabled' in rendered_contents[0]


def test_run_script_with_context_temp_file_cleanup(tmp_path, monkeypatch):
    """Test that temporary files are created with correct extension."""
    script_content = "#!/usr/bin/env python\nprint('{{ test }}')"
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding='utf-8')
    
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    context = {'test': 'value'}
    
    created_temp_files = []
    
    def mock_run_script_impl(script_path, cwd):
        created_temp_files.append(script_path)
        # Verify temp file has .py extension
        assert script_path.endswith('.py')
    
    monkeypatch.setattr('cookiecutter.hooks.run_script', mock_run_script_impl)
    
    run_script_with_context(str(script_path), cwd, context)
    
    assert len(created_temp_files) == 1


