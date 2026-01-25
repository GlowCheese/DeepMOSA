####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create a temporary script file
    script_path = os.path.join(temp_dir, 'test_script.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('print("Hello, World!")')

    # Define context
    context = {'project_name': 'Test Project'}

    # Run the script with context
    run_script_with_context(script_path, temp_dir, context)

    # Clean up
    rmtree(temp_dir)


# LLM-generated content at query #2
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: Valid hook directory with a valid hook script
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('pre_gen_project.py')
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.rmdir(hooks_dir)

    # Test case 2: Valid hook directory with no valid hook script
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'random_script.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, 'random_script.py'))
    os.rmdir(hooks_dir)

    # Test case 3: Invalid hook directory
    hooks_dir = 'invalid_hooks'
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None

    # Test case 4: Hook directory with backup file
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py~'))
    os.rmdir(hooks_dir)

    # Test case 5: Hook directory with unsupported hook
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'unsupported_hook.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('unsupported_hook', hooks_dir)
    assert result is None
    os.remove(os.path.join(hooks_dir, 'unsupported_hook.py'))
    os.rmdir(hooks_dir)

    # Test case 6: Hook directory with multiple valid hook scripts
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
        f.write('echo "Hello, World!"')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 2
    assert any(path.endswith('pre_gen_project.py') for path in result)
    assert any(path.endswith('pre_gen_project.sh') for path in result)
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.sh'))
    os.rmdir(hooks_dir)

    # Test case 7: Hook directory with a valid hook script and a non-hook script
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    with open(os.path.join(hooks_dir, 'random_script.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('pre_gen_project.py')
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.remove(os.path.join(hooks_dir, 'random_script.py'))
    os.rmdir(hooks_dir)

    # Test case 8: Hook directory with a valid hook script and a backup file
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    with open(os.path.join(hooks_dir, 'pre_gen_project.py~'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('pre_gen_project.py')
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py~'))
    os.rmdir(hooks_dir)

    # Test case 9: Hook directory with a valid hook script and an unsupported hook
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    with open(os.path.join(hooks_dir, 'unsupported_hook.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 1
    assert result[0].endswith('pre_gen_project.py')
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.remove(os.path.join(hooks_dir, 'unsupported_hook.py'))
    os.rmdir(hooks_dir)

    # Test case 10: Hook directory with multiple valid hook scripts and non-hook scripts
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Hello, World!")')
    with open(os.path.join(hooks_dir, 'pre_gen_project.sh'), 'w') as f:
        f.write('echo "Hello, World!"')
    with open(os.path.join(hooks_dir, 'random_script.py'), 'w') as f:
        f.write('print("Hello, World!")')
    result = find_hook('pre_gen_project', hooks_dir)
    assert result is not None
    assert len(result) == 2
    assert any(path.endswith('pre_gen_project.py') for path in result)
    assert any(path.endswith('pre_gen_project.sh') for path in result)
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.py'))
    os.remove(os.path.join(hooks_dir, 'pre_gen_project.sh'))
    os.remove(os.path.join(hooks_dir, 'random_script.py'))
    os.rmdir(hooks_dir)


# LLM-generated content at query #3
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    repo_dir = Path("test_repo")
    hook_name = "test_hook"
    project_dir = Path("test_project")
    context = {"key": "value"}
    delete_project_on_failure = True

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert not project_dir.exists(), "Project directory should be deleted on hook failure"
    else:
        assert True, "Hook should execute successfully"



# LLM-generated content at query #4
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test the run_hook_from_repo_dir function."""
    # Setup temporary directories and files
    with tempfile.TemporaryDirectory() as tmp_repo_dir:
        hooks_dir = os.path.join(tmp_repo_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello from hook")')

        project_dir = os.path.join(tmp_repo_dir, 'project')
        os.makedirs(project_dir)

        context = {'cookiecutter': {'project_name': 'Test Project'}}

        # Test successful hook execution
        run_hook_from_repo_dir(tmp_repo_dir, 'pre_gen_project', project_dir, context, True)

        # Test hook failure with delete_project_on_failure=True
        with open(hook_file, 'w') as f:
            f.write('exit(1)')
        try:
            run_hook_from_repo_dir(tmp_repo_dir, 'pre_gen_project', project_dir, context, True)
        except FailedHookException:
            assert not os.path.exists(project_dir), "Project directory should be deleted on hook failure"

        # Test hook failure with delete_project_on_failure=False
        os.makedirs(project_dir)
        with open(hook_file, 'w') as f:
            f.write('exit(1)')
        try:
            run_hook_from_repo_dir(tmp_repo_dir, 'pre_gen_project', project_dir, context, False)
        except FailedHookException:
            assert os.path.exists(project_dir), "Project directory should not be deleted on hook failure"


# LLM-generated content at query #5
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir)
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        script_path = hooks_dir / "pre_prompt.py"
        script_path.write_text("print('Pre-prompt hook executed')")

        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

        script_path.unlink()
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

        invalid_script_path = hooks_dir / "invalid_pre_prompt.py"
        invalid_script_path.write_text("")

        try:
            run_pre_prompt_hook(repo_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException"


# LLM-generated content at query #6
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Test that the script is executed correctly
    script_path = Path("test_script.py")
    cwd = Path(".")
    context = {"test": "test"}
    try:
        run_script_with_context(script_path, cwd, context)
    except FailedHookException:
        pass


# LLM-generated content at query #7
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    """Test the find_hook function."""
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project') is None

    # Test case 2: Hooks directory exists but no matching hook
    os.makedirs('hooks', exist_ok=True)
    assert find_hook('pre_gen_project') is None

    # Test case 3: Hooks directory exists with matching hook
    with open('hooks/pre_gen_project.py', 'w', encoding='utf-8') as f:
        f.write('print("Hello, World!")')
    assert find_hook('pre_gen_project') is not None

    # Test case 4: Hooks directory exists with backup file
    with open('hooks/pre_gen_project.py~', 'w', encoding='utf-8') as f:
        f.write('print("Hello, World!")')
    assert find_hook('pre_gen_project') is not None

    # Clean up
    os.remove('hooks/pre_gen_project.py')
    os.remove('hooks/pre_gen_project.py~')
    os.rmdir('hooks')


# LLM-generated content at query #8
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script file
        script_path = os.path.join(tmp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, world!")')
        
        # Test running the script
        run_script(script_path, tmp_dir)
        
        # Test running a non-existent script
        try:
            run_script(os.path.join(tmp_dir, 'nonexistent.py'), tmp_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException for non-existent script"
        
        # Test running an empty script
        empty_script_path = os.path.join(tmp_dir, 'empty_script.py')
        with open(empty_script_path, 'w') as f:
            pass
        try:
            run_script(empty_script_path, tmp_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException for empty script"


# LLM-generated content at query #9
#--------------------------

# Unit test for function valid_hook
def test_valid_hook():
    assert valid_hook('pre_prompt.py', 'pre_prompt') == True
    assert valid_hook('pre_prompt.sh', 'pre_prompt') == True
    assert valid_hook('pre_prompt.py~', 'pre_prompt') == False
    assert valid_hook('post_gen_project.py', 'pre_prompt') == False
    assert valid_hook('invalid_hook.py', 'pre_prompt') == False



# LLM-generated content at query #10
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Test case 1: Successful hook execution
    repo_dir = "/path/to/repo"
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"key": "value"}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    # Assert that the project directory is not deleted (since hook succeeded)

    # Test case 2: Failed hook execution
    repo_dir = "/path/to/repo"
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"key": "value"}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        # Assert that the project directory is deleted
        assert not os.path.exists(project_dir)

    # Test case 3: Failed hook execution with delete_project_on_failure set to False
    repo_dir = "/path/to/repo"
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"key": "value"}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        # Assert that the project directory is not deleted
        assert os.path.exists(project_dir)


# LLM-generated content at query #11
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook script
    repo_dir = "path/to/repo"
    assert run_pre_prompt_hook(repo_dir) == repo_dir

    # Test case 2: Valid pre_prompt hook script, no exception
    repo_dir = "path/to/repo_with_pre_prompt_hook"
    assert run_pre_prompt_hook(repo_dir) != repo_dir

    # Test case 3: Valid pre_prompt hook script, raises FailedHookException
    repo_dir = "path/to/repo_with_failing_pre_prompt_hook"
    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException:
        pass
    else:
        assert False, "Expected FailedHookException"


# LLM-generated content at query #12
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Test when there is no pre_prompt hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_pre_prompt_hook(tmp_dir)
        assert result == tmp_dir

    # Test when there is a pre_prompt hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_script = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_script, 'w') as f:
            f.write('print("Pre-prompt hook executed")')
        result = run_pre_prompt_hook(tmp_dir)
        assert isinstance(result, str)

    # Test when pre_prompt hook fails
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_script = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_script, 'w') as f:
            f.write('exit(1)')
        try:
            run_pre_prompt_hook(tmp_dir)
        except FailedHookException:
            assert True
        else:
            assert False


# LLM-generated content at query #13
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    # Mock context and project directory
    context = {'project_name': 'test_project'}
    project_dir = '/path/to/project'

    # Test with a valid hook
    # Assuming 'pre_gen_project' hook exists in the template
    try:
        run_hook('pre_gen_project', project_dir, context)
    except FailedHookException:
        assert False, "run_hook failed with a valid hook"

    # Test with an invalid hook
    try:
        run_hook('invalid_hook', project_dir, context)
    except FailedHookException:
        assert False, "run_hook failed with an invalid hook"

    # Test with no hooks directory
    try:
        run_hook('pre_gen_project', project_dir, context)
    except FailedHookException:
        assert False, "run_hook failed with no hooks directory"

    # Test with a failing hook script
    # Assuming 'post_gen_project' hook exists and fails
    try:
        run_hook('post_gen_project', project_dir, context)
    except FailedHookException:
        pass  # Expected to fail

    # Test with a hook script that throws an OSError
    # Assuming 'post_gen_project' hook exists and throws OSError
    try:
        run_hook('post_gen_project', project_dir, context)
    except FailedHookException:
        pass  # Expected to fail

    # Test with a hook script that throws an UndefinedError
    # Assuming 'post_gen_project' hook exists and throws UndefinedError
    try:
        run_hook('post_gen_project', project_dir, context)
    except FailedHookException:
        pass  # Expected to fail

    print("All tests passed.")



# LLM-generated content at query #14
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir function."""
    # Setup test environment
    test_repo_dir = Path(tempfile.mkdtemp())
    test_project_dir = Path(tempfile.mkdtemp())
    test_context = {'project_name': 'test_project'}

    # Create a test hook script
    hook_script = test_repo_dir / 'hooks' / 'pre_gen_project.sh'
    hook_script.parent.mkdir(exist_ok=True)
    hook_script.write_text('#!/bin/bash\necho "Running pre_gen_project hook"')

    # Test successful hook execution
    try:
        run_hook_from_repo_dir(
            test_repo_dir,
            'pre_gen_project',
            test_project_dir,
            test_context,
            True,
        )
    except Exception as e:
        assert False, f"run_hook_from_repo_dir raised an exception: {e}"

    # Test hook failure with delete_project_on_failure=True
    hook_script.write_text('#!/bin/bash\nexit 1')  # Script that fails
    try:
        run_hook_from_repo_dir(
            test_repo_dir,
            'pre_gen_project',
            test_project_dir,
            test_context,
            True,
        )
    except FailedHookException:
        assert not test_project_dir.exists(), "Project directory should be deleted"
    else:
        assert False, "FailedHookException not raised"

    # Cleanup
    rmtree(test_repo_dir)
    if test_project_dir.exists():
        rmtree(test_project_dir)


# LLM-generated content at query #15
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test the run_script_with_context function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary script file
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, {{ cookiecutter.name }}!")')

        # Create a context
        context = {'cookiecutter': {'name': 'World'}}

        # Run the script
        run_script_with_context(script_path, temp_dir, context)

        # Check that the script ran successfully
        assert True

    # Test with a non-existent script
    try:
        run_script_with_context('non_existent_script.py', temp_dir, context)
    except FailedHookException:
        assert True
    else:
        assert False

    # Test with a script that raises an error
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('raise ValueError("Error")')

        try:
            run_script_with_context(script_path, temp_dir, context)
        except FailedHookException:
            assert True
        else:
            assert False


# LLM-generated content at query #16
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    """Test the run_pre_prompt_hook function."""
    # Create a temporary directory to simulate a repo with a pre_prompt hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a hooks directory and a pre_prompt script
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        script_path = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Running pre_prompt hook")')

        # Test that the function runs the pre_prompt hook
        result = run_pre_prompt_hook(tmp_dir)
        assert os.path.isdir(result)  # Should return a directory path

    # Test with a repo that has no hooks directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_pre_prompt_hook(tmp_dir)
        assert result == tmp_dir  # Should return the same directory

    # Test with a repo that has an invalid pre_prompt hook (empty file)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        script_path = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            pass  # Empty file

        try:
            run_pre_prompt_hook(tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass  # Expected

    # Test with a repo that has a failing pre_prompt hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        script_path = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('import sys; sys.exit(1)')  # Script that exits with error

        try:
            run_pre_prompt_hook(tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass  # Expected


# LLM-generated content at query #17
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    repo_dir = Path("fake_repo_dir")
    hook_name = "pre_gen_project"
    project_dir = Path("fake_project_dir")
    context = {"key": "value"}
    delete_project_on_failure = True

    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert True
    except Exception as e:
        assert False, f"Unexpected exception: {e}"
    else:
        assert False, "Expected FailedHookException to be raised"


# LLM-generated content at query #18
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Test case for successful script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as temp_file:
        temp_file.write("#!/bin/bash\n")
        temp_file.write("exit 0\n")
        temp_file.flush()
        run_script(temp_file.name)

    # Test case for failed script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as temp_file:
        temp_file.write("#!/bin/bash\n")
        temp_file.write("exit 1\n")
        temp_file.flush()
        try:
            run_script(temp_file.name)
        except FailedHookException as e:
            assert str(e) == "Hook script failed (exit status: 1)"

    # Test case for empty file or missing shebang
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as temp_file:
        temp_file.write("")
        temp_file.flush()
        try:
            run_script(temp_file.name)
        except FailedHookException as e:
            assert str(e) == "Hook script failed, might be an empty file or missing a shebang"

    # Test case for other OSError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as temp_file:
        temp_file.write("#!/bin/bash\n")
        temp_file.write("exit 0\n")
        temp_file.flush()
        try:
            run_script("/nonexistent/path/to/script.sh")
        except FailedHookException as e:
            assert str(e).startswith("Hook script failed (error: ")



# LLM-generated content at query #19
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock pre_prompt hook script
        script_path = os.path.join(temp_dir, 'pre_prompt.py')
        with open(script_path, 'w') as f:
            f.write('print("Pre-Prompt Hook Executed")')
        
        # Test the function
        result_dir = run_pre_prompt_hook(temp_dir)
        
        # Assert that the function returns the correct directory
        assert result_dir == temp_dir


# LLM-generated content at query #20
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test the run_script_with_context function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a script file
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, {{ name }}!")')

        # Create a context
        context = {'name': 'World'}

        # Run the script
        run_script_with_context(script_path, temp_dir, context)

        # Check the output
        with open(os.path.join(temp_dir, 'test_script.py'), 'r', encoding='utf-8') as f:
            assert f.read() == 'print("Hello, World!")'


# LLM-generated content at query #21
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Mock the find_hook function to return a script path
    original_find_hook = find_hook
    find_hook = lambda hook_name, hooks_dir='hooks': ['/path/to/script']

    # Mock the run_script_with_context function
    original_run_script_with_context = run_script_with_context
    run_script_with_context = lambda script_path, cwd, context: None

    # Test with a valid hook
    run_hook('pre_gen_project', '/project/dir', {'key': 'value'})

    # Test with no hook found
    find_hook = lambda hook_name, hooks_dir='hooks': None
    run_hook('pre_gen_project', '/project/dir', {'key': 'value'})

    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context


# LLM-generated content at query #22
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Valid hooks directory with no matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 3: Valid hooks directory with matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('')
        assert find_hook('pre_gen_project', hooks_dir) == [hook_file]

    # Test case 4: Valid hooks directory with backup file
    with tempfile.TemporaryDirectory() as temp_dir:
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('')
        assert find_hook('pre_gen_project', hooks_dir) is None



# LLM-generated content at query #23
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Create a valid hook file
        valid_hook_file = hooks_dir / 'pre_gen_project.sh'
        valid_hook_file.write_text('#!/bin/bash\necho "Hello World"')
        
        # Create an invalid hook file
        invalid_hook_file = hooks_dir / 'invalid_hook.sh'
        invalid_hook_file.write_text('#!/bin/bash\necho "Invalid Hook"')
        
        # Test valid hook
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result == [str(valid_hook_file)]
        
        # Test invalid hook
        result = find_hook('invalid_hook', str(hooks_dir))
        assert result is None
        
        # Test non-existent hook
        result = find_hook('post_gen_project', str(hooks_dir))
        assert result is None


# LLM-generated content at query #24
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Create a temporary directory to simulate a repo directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock pre_prompt hook script
        hook_script_path = Path(tmp_dir) / 'hooks' / 'pre_prompt.py'
        hook_script_path.parent.mkdir(parents=True, exist_ok=True)
        hook_script_path.write_text('print("Pre-prompt hook executed")')

        # Test the function with the mock repo directory
        result = run_pre_prompt_hook(tmp_dir)

        # Assert that the function returns the correct directory
        assert result == tmp_dir

        # Test the function with a non-existing hook
        hook_script_path.unlink()
        result = run_pre_prompt_hook(tmp_dir)
        assert result == tmp_dir

        # Test the function with a failing hook
        failing_hook_script_path = Path(tmp_dir) / 'hooks' / 'pre_prompt.py'
        failing_hook_script_path.write_text('import sys; sys.exit(1)')
        try:
            run_pre_prompt_hook(tmp_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException"

        # Clean up
        rmtree(tmp_dir)


# LLM-generated content at query #25
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Test case 1: No pre_prompt hook in the repo directory
    repo_dir = "/path/to/repo"
    assert run_pre_prompt_hook(repo_dir) == repo_dir

    # Test case 2: Valid pre_prompt hook in the repo directory
    repo_dir = "/path/to/repo_with_hook"
    assert run_pre_prompt_hook(repo_dir) != repo_dir

    # Test case 3: Pre_prompt hook fails
    repo_dir = "/path/to/repo_with_failing_hook"
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        assert True


# LLM-generated content at query #26
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary script file
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, World!")')

        # Define a context dictionary
        context = {'project_name': 'TestProject'}

        # Call the function
        run_script_with_context(script_path, temp_dir, context)

        # Ensure the script executed successfully
        assert True  # No exception means the script ran successfully



# LLM-generated content at query #27
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Creating a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write('#!/bin/sh\necho "Hello, World!"\n')
        script_path = temp.name

    # Making the script executable
    os.chmod(script_path, 0o755)

    # Running the script
    try:
        run_script(script_path)
    finally:
        # Cleaning up the temporary script file
        os.unlink(script_path)



# LLM-generated content at query #28
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Mock the necessary dependencies
    import pytest
    from unittest.mock import patch
    
    # Test case 1: No pre_prompt hook found
    repo_dir = "/fake/repo/dir"
    with patch("cookiecutter.hooks.find_hook", return_value=None):
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
    
    # Test case 2: Valid pre_prompt hook found and executed successfully
    mock_script = "/fake/repo/dir/hooks/pre_prompt.sh"
    with patch("cookiecutter.hooks.find_hook", return_value=[mock_script]), \
         patch("cookiecutter.hooks.run_script") as mock_run_script, \
         patch("cookiecutter.utils.create_tmp_repo_dir", return_value=repo_dir):
        result = run_pre_prompt_hook(repo_dir)
        mock_run_script.assert_called_with(mock_script, repo_dir)
        assert result == repo_dir
    
    # Test case 3: Pre_prompt hook fails
    with patch("cookiecutter.hooks.find_hook", return_value=[mock_script]), \
         patch("cookiecutter.hooks.run_script", side_effect=FailedHookException("Hook failed")), \
         patch("cookiecutter.utils.create_tmp_repo_dir", return_value=repo_dir):
        with pytest.raises(FailedHookException):
            run_pre_prompt_hook(repo_dir)


# LLM-generated content at query #29
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script file
        script_path = os.path.join(tmp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')
        
        # Test running the script
        run_script(script_path, tmp_dir)
        
        # Test with a non-existent script
        try:
            run_script(os.path.join(tmp_dir, 'nonexistent.py'), tmp_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException for non-existent script"


# LLM-generated content at query #30
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    hooks_dir = 'nonexistent_directory'
    assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 2: Valid hooks directory with no matching hook
    hooks_dir = 'hooks'
    os.makedirs(hooks_dir, exist_ok=True)
    assert find_hook('invalid_hook', hooks_dir) is None

    # Test case 3: Valid hooks directory with matching hook
    hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file, 'w') as f:
        f.write('')
    assert find_hook('pre_gen_project', hooks_dir) == [os.path.abspath(hook_file)]

    # Clean up
    os.remove(hook_file)
    os.rmdir(hooks_dir)



# LLM-generated content at query #31
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Mock data
    script_path = "test_script.py"
    cwd = "."

    # Prepare the test environment
    with open(script_path, "w") as f:
        f.write("print('Hello, World!')")
    utils.make_executable(script_path)

    # Execute the function
    run_script(script_path, cwd)

    # Clean up
    os.remove(script_path)



# LLM-generated content at query #32
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Test successful script execution
    import tempfile
    import os
    test_script_content = "#!/bin/bash\n\necho 'Hello, World!'"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write(test_script_content)
        tmp_file_path = tmp_file.name
    os.chmod(tmp_file_path, 0o755)
    run_script(tmp_file_path)
    os.unlink(tmp_file_path)

    # Test script execution failure
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("#!/bin/bash\n\nexit 1")
        tmp_file_path = tmp_file.name
    os.chmod(tmp_file_path, 0o755)
    try:
        run_script(tmp_file_path)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    os.unlink(tmp_file_path)

    # Test script execution with non-executable file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("#!/bin/bash\n\necho 'Hello, World!'")
        tmp_file_path = tmp_file.name
    try:
        run_script(tmp_file_path)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    os.unlink(tmp_file_path)


# LLM-generated content at query #33
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir function."""
    # Setup
    test_repo_dir = Path(tempfile.mkdtemp())
    test_project_dir = Path(tempfile.mkdtemp())
    test_context = {'project_name': 'test_project'}
    test_hook_name = 'pre_gen_project'
    test_delete_on_failure = True

    # Create a test hook script
    hook_script = test_repo_dir / 'hooks' / 'pre_gen_project.py'
    hook_script.parent.mkdir(exist_ok=True)
    hook_script.write_text('print("Running pre_gen_project hook")')

    # Test successful execution
    try:
        run_hook_from_repo_dir(
            test_repo_dir,
            test_hook_name,
            test_project_dir,
            test_context,
            test_delete_on_failure,
        )
    except Exception as e:
        assert False, f"run_hook_from_repo_dir raised an exception: {e}"

    # Cleanup
    rmtree(test_repo_dir)
    rmtree(test_project_dir)


# LLM-generated content at query #34
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Setup: Create a temporary script file with Jinja2 placeholders
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_script:
        temp_script.write("echo '{{ greeting }}'")
        script_path = temp_script.name

    # Setup: Define a context with a greeting message
    context = {"greeting": "Hello, World!"}

    # Setup: Create a temporary directory to run the script in
    with tempfile.TemporaryDirectory() as temp_dir:
        # Execute: Run the script with the context
        run_script_with_context(script_path, temp_dir, context)

    # Clean up the script file
    os.unlink(script_path)


# LLM-generated content at query #35
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Mock repo directory and pre_prompt hook
    repo_dir = Path(tempfile.mkdtemp())
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "pre_prompt.py"
    hook_script.write_text("print('Pre-prompt hook executed')")

    # Test successful execution
    try:
        updated_repo_dir = run_pre_prompt_hook(repo_dir)
        assert updated_repo_dir == repo_dir
    finally:
        rmtree(repo_dir)

    # Test execution failure
    hook_script.write_text("raise Exception('Hook failed')")
    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    finally:
        rmtree(repo_dir)


# LLM-generated content at query #36
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Valid hook directory but no matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        assert find_hook('pre_gen_project', temp_dir) is None

    # Test case 3: Valid hook directory with matching hook
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', temp_dir) == [os.path.abspath(hook_file)]

    # Test case 4: Multiple hooks in directory
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file1 = os.path.join(temp_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(temp_dir, 'post_gen_project.py')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        assert find_hook('pre_gen_project', temp_dir) == [os.path.abspath(hook_file1)]

    # Test case 5: Invalid hook file (backup file)
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', temp_dir) is None



# LLM-generated content at query #37
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Setup
    test_repo_dir = Path(tempfile.mkdtemp())
    
    # Test case 1: No pre_prompt script
    assert run_pre_prompt_hook(test_repo_dir) == test_repo_dir
    
    # Test case 2: Valid pre_prompt script
    hooks_dir = test_repo_dir / 'hooks'
    hooks_dir.mkdir()
    script_path = hooks_dir / 'pre_prompt.py'
    script_path.write_text('print("Running pre_prompt hook")')
    
    result = run_pre_prompt_hook(test_repo_dir)
    assert result != test_repo_dir
    
    # Cleanup
    rmtree(test_repo_dir)


# LLM-generated content at query #38
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test run_script_with_context function."""
    # Create a temporary file with a Jinja template
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as temp:
        temp.write('echo {{ message }}')

    # Define context and working directory
    context = {'message': 'Hello, World!'}
    cwd = Path(tempfile.mkdtemp())

    # Run the script
    run_script_with_context(temp.name, cwd, context)

    # Clean up
    os.unlink(temp.name)
    rmtree(cwd)


# LLM-generated content at query #39
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    # Mock context and project directory
    context = {'project_name': 'test_project'}
    project_dir = Path('/tmp/test_project')

    # Mock script path
    script_path = '/tmp/hooks/pre_gen_project.py'

    # Mock find_hook to return the script path
    original_find_hook = find_hook
    find_hook = lambda hook_name, hooks_dir='hooks': [script_path]

    # Mock run_script_with_context
    original_run_script_with_context = run_script_with_context
    run_script_with_context = lambda script_path, cwd, context: None

    # Run the hook
    run_hook('pre_gen_project', project_dir, context)

    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context



# LLM-generated content at query #40
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Mock the find_hook function to return a script path
    original_find_hook = find_hook
    find_hook = lambda hook_name, hooks_dir='hooks': ['/path/to/script']  # noqa: E731

    # Mock the run_script_with_context function
    original_run_script_with_context = run_script_with_context
    run_script_with_context = lambda script_path, cwd, context: None  # noqa: E731

    # Test with a valid hook name and context
    try:
        run_hook('pre_gen_project', '.', {'key': 'value'})
    except Exception as e:
        assert False, f"run_hook raised an exception: {e}"

    # Test with no scripts found
    find_hook = lambda hook_name, hooks_dir='hooks': None  # noqa: E731
    try:
        run_hook('pre_gen_project', '.', {'key': 'value'})
    except Exception as e:
        assert False, f"run_hook raised an exception: {e}"

    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context


# LLM-generated content at query #41
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script file
        script_path = os.path.join(tmp_dir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\necho "Hello, World!"\n')
        os.chmod(script_path, 0o755)

        # Test running the script
        run_script(script_path, tmp_dir)

        # Test running a non-existent script
        try:
            run_script(os.path.join(tmp_dir, 'nonexistent.sh'), tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass

        # Test running a script with an error
        error_script_path = os.path.join(tmp_dir, 'error_script.sh')
        with open(error_script_path, 'w') as f:
            f.write('#!/bin/bash\nexit 1\n')
        os.chmod(error_script_path, 0o755)
        try:
            run_script(error_script_path, tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass


# LLM-generated content at query #42
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Setup: Create a temporary directory and a pre_prompt hook script
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        hook_script = repo_dir / "hooks" / "pre_prompt.py"
        hook_script.parent.mkdir()
        hook_script.write_text("print('Running pre_prompt hook')")

        # Test: Run the pre_prompt hook
        result = run_pre_prompt_hook(repo_dir)

        # Verify: Ensure the hook was executed successfully
        assert result == repo_dir

    # Test: Run pre_prompt hook without a hooks directory
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir

    # Test: Run pre_prompt hook with a failing script
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        hook_script = repo_dir / "hooks" / "pre_prompt.py"
        hook_script.parent.mkdir()
        hook_script.write_text("import sys; sys.exit(1)")

        try:
            run_pre_prompt_hook(repo_dir)
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e.__cause__)


# LLM-generated content at query #43
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Create a temporary directory to simulate a repo directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock pre_prompt script
        script_path = Path(tmp_dir) / "hooks" / "pre_prompt.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('Pre-prompt hook executed')")

        # Run the hook
        result = run_pre_prompt_hook(tmp_dir)

        # Assert that the result is the same as the input directory
        assert result == tmp_dir

        # Clean up
        rmtree(tmp_dir)


# LLM-generated content at query #44
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Setup
    test_script_path = "test_script.py"
    test_dir = "test_dir"
    
    # Mock subprocess.Popen and utils.make_executable
    def mock_popen(*args, **kwargs):
        class MockProcess:
            def wait(self):
                return EXIT_SUCCESS
        return MockProcess()
    
    original_popen = subprocess.Popen
    subprocess.Popen = mock_popen
    
    def mock_make_executable(path):
        pass
    
    original_make_executable = utils.make_executable
    utils.make_executable = mock_make_executable
    
    # Test successful execution
    try:
        run_script(test_script_path, test_dir)
    except FailedHookException as e:
        assert False, f"run_script raised an exception: {e}"
    
    # Test failed execution
    def mock_popen_fail(*args, **kwargs):
        class MockProcess:
            def wait(self):
                return 1
        return MockProcess()
    
    subprocess.Popen = mock_popen_fail
    
    try:
        run_script(test_script_path, test_dir)
        assert False, "run_script did not raise FailedHookException for failed execution"
    except FailedHookException:
        pass
    
    # Test OSError handling
    def mock_popen_oserror(*args, **kwargs):
        raise OSError(errno.ENOEXEC, "Test OSError")
    
    subprocess.Popen = mock_popen_oserror
    
    try:
        run_script(test_script_path, test_dir)
        assert False, "run_script did not raise FailedHookException for OSError"
    except FailedHookException:
        pass
    
    # Restore original functions
    subprocess.Popen = original_popen
    utils.make_executable = original_make_executable
    
    # Cleanup
    os.remove(test_script_path)
    os.rmdir(test_dir)


# LLM-generated content at query #45
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'non_existent_dir') is None

    # Test case 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as temp_dir:
        assert find_hook('pre_gen_project', temp_dir) is None

    # Test case 3: Valid hook file
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello, World!")')
        assert find_hook('pre_gen_project', temp_dir) == [os.path.abspath(hook_file)]

    # Test case 4: Invalid hook file (backup file)
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('print("Hello, World!")')
        assert find_hook('pre_gen_project', temp_dir) is None

    # Test case 5: Invalid hook file (unsupported hook)
    with tempfile.TemporaryDirectory() as temp_dir:
        hook_file = os.path.join(temp_dir, 'unsupported_hook.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello, World!")')
        assert find_hook('pre_gen_project', temp_dir) is None


# LLM-generated content at query #46
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script
        script_path = os.path.join(tmp_dir, 'test_script.sh')
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\necho "Hello, World!"')
        # Run the script
        run_script(script_path, tmp_dir)
        # Check if the script was executed successfully
        assert True



# LLM-generated content at query #47
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    repo_dir = Path('test_repo')
    hook_name = 'pre_gen_project'
    project_dir = Path('test_project')
    context = {'project_name': 'test_project'}
    delete_project_on_failure = True

    # Create temporary repo directory and hooks directory
    repo_dir.mkdir(exist_ok=True)
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir(exist_ok=True)

    # Create a valid pre_gen_project hook script
    script_path = hooks_dir / 'pre_gen_project.py'
    script_path.write_text('print("Running pre_gen_project hook")')

    # Run the hook
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    # Clean up
    rmtree(repo_dir)
    if project_dir.exists():
        rmtree(project_dir)


# LLM-generated content at query #48
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        hooks_dir = repo_dir / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'post_gen_project.py'
        hook_script.write_text('import sys; sys.exit(0)')
        project_dir = Path(tmpdir) / 'project'
        project_dir.mkdir()
        context = {}
        run_hook_from_repo_dir(repo_dir, 'post_gen_project', project_dir, context, True)


# LLM-generated content at query #49
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """
    Test run_hook function.
    """
    # Mock data
    hook_name = "pre_gen_project"
    project_dir = "/tmp/test_project"
    context = {"project_name": "Test Project"}
    
    # Test case 1: No hook found
    # Mock find_hook to return None
    def find_hook_mock(hook_name, hooks_dir='hooks'):
        return None
    
    original_find_hook = find_hook
    find_hook = find_hook_mock
    
    run_hook(hook_name, project_dir, context)
    
    # Restore original find_hook
    find_hook = original_find_hook
    
    # Test case 2: Hook found and executed successfully
    # Mock find_hook to return a script path
    def find_hook_mock(hook_name, hooks_dir='hooks'):
        return ["/tmp/test_project/hooks/pre_gen_project.py"]
    
    # Mock run_script_with_context to do nothing
    def run_script_with_context_mock(script_path, cwd, context):
        pass
    
    original_find_hook = find_hook
    find_hook = find_hook_mock
    original_run_script_with_context = run_script_with_context
    run_script_with_context = run_script_with_context_mock
    
    run_hook(hook_name, project_dir, context)
    
    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context
    
    # Test case 3: Hook execution fails
    # Mock find_hook to return a script path
    def find_hook_mock(hook_name, hooks_dir='hooks'):
        return ["/tmp/test_project/hooks/pre_gen_project.py"]
    
    # Mock run_script_with_context to raise FailedHookException
    def run_script_with_context_mock(script_path, cwd, context):
        raise FailedHookException("Hook script failed")
    
    original_find_hook = find_hook
    find_hook = find_hook_mock
    original_run_script_with_context = run_script_with_context
    run_script_with_context = run_script_with_context_mock
    
    try:
        run_hook(hook_name, project_dir, context)
    except FailedHookException:
        pass
    
    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Implement test cases here
    pass


# LLM-generated content at query #2
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert find_hook('pre_gen_project', tmp_dir) is None

    # Test case 3: Valid hook file
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello, world!")')
        assert find_hook('pre_gen_project', hook_dir) == [os.path.abspath(hook_file)]

    # Test case 4: Invalid hook file (backup file)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('print("Hello, world!")')
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 5: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'pre_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('print("Hello, world!")')
        with open(hook_file2, 'w') as f:
            f.write('echo "Hello, world!"')
        assert sorted(find_hook('pre_gen_project', hook_dir)) == sorted([
            os.path.abspath(hook_file1),
            os.path.abspath(hook_file2),
        ])


# LLM-generated content at query #3
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Mock script path and context
    script_path = Path("/tmp/test_script.py")
    context = {"project_name": "test_project"}
    cwd = Path("/tmp")

    # Create a dummy script file
    script_path.write_text("print('{{ project_name }}')", encoding="utf-8")

    # Mock subprocess.Popen and utils.make_executable
    original_popen = subprocess.Popen
    original_make_executable = utils.make_executable

    def mock_popen(*args, **kwargs):
        return MockProcess()

    class MockProcess:
        def wait(self):
            return 0

    subprocess.Popen = mock_popen
    utils.make_executable = lambda x: None

    try:
        # Run the function
        run_script_with_context(script_path, cwd, context)
    finally:
        # Restore original functions
        subprocess.Popen = original_popen
        utils.make_executable = original_make_executable
        # Clean up the temporary script
        script_path.unlink()


# LLM-generated content at query #4
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    repo_dir = Path('test_repo_dir')
    hook_name = 'test_hook'
    project_dir = Path('test_project_dir')
    context = {'test_key': 'test_value'}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert True  # Placeholder assertion


# LLM-generated content at query #5
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Setup: Create a temporary directory and a dummy pre_prompt hook script
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_script = os.path.join(hooks_dir, 'pre_prompt.py')
        with open(hook_script, 'w') as f:
            f.write("print('Running pre_prompt hook')")

        # Test: Ensure the function runs the hook script
        try:
            result = run_pre_prompt_hook(tmpdir)
            assert result == tmpdir, "Expected the same directory to be returned"
        except FailedHookException as e:
            assert False, f"run_pre_prompt_hook failed with {e}"

        # Test: Ensure the function handles non-existent hooks correctly
        os.remove(hook_script)
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir, "Expected the same directory to be returned when no hook exists"


# LLM-generated content at query #6
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Create a temporary directory with a pre_prompt hook script
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = Path(tmp_dir) / 'hooks'
        hooks_dir.mkdir()
        hook_script = hooks_dir / 'pre_prompt.py'
        hook_script.write_text('print("pre_prompt hook executed")')

        # Run the pre_prompt hook
        result = run_pre_prompt_hook(tmp_dir)

        # Verify that the function returns the correct directory
        assert result == tmp_dir


# LLM-generated content at query #7
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp:
        temp.write("#!/bin/bash\necho 'Hello, World!'")
        script_path = temp.name

    # Run the script
    try:
        run_script(script_path)
    finally:
        os.remove(script_path)



# LLM-generated content at query #8
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: Directory with hooks
    hooks_dir = "hooks"
    hook_name = "pre_gen_project"
    # Create hooks directory and a valid hook file
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_gen_project.py"), "w") as f:
        f.write("")
    assert find_hook(hook_name, hooks_dir) is not None
    # Clean up
    os.remove(os.path.join(hooks_dir, "pre_gen_project.py"))
    os.rmdir(hooks_dir)

    # Test case 2: Directory without hooks
    hooks_dir = "no_hooks"
    hook_name = "pre_gen_project"
    os.makedirs(hooks_dir, exist_ok=True)
    assert find_hook(hook_name, hooks_dir) is None
    os.rmdir(hooks_dir)

    # Test case 3: Invalid hook name
    hooks_dir = "hooks"
    hook_name = "invalid_hook"
    os.makedirs(hooks_dir, exist_ok=True)
    assert find_hook(hook_name, hooks_dir) is None
    os.rmdir(hooks_dir)

    # Test case 4: Backup file
    hooks_dir = "hooks"
    hook_name = "pre_gen_project"
    os.makedirs(hooks_dir, exist_ok=True)
    with open(os.path.join(hooks_dir, "pre_gen_project.py~"), "w") as f:
        f.write("")
    assert find_hook(hook_name, hooks_dir) is None
    os.remove(os.path.join(hooks_dir, "pre_gen_project.py~"))
    os.rmdir(hooks_dir)


# LLM-generated content at query #9
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Mocking necessary objects and contexts
    mock_repo_dir = Path('/mock/repo_dir')
    mock_project_dir = Path('/mock/project_dir')
    mock_context = {'key': 'value'}
    
    # Test case 1: Hook script exists and runs successfully
    # Mock find_hook to return a script
    # Mock run_script_with_context to not raise any exception
    # Assert that no exception is raised
    
    # Test case 2: Hook script exists but raises FailedHookException
    # Mock find_hook to return a script
    # Mock run_script_with_context to raise FailedHookException
    # Assert that FailedHookException is raised and project_dir is deleted if delete_project_on_failure is True
    
    # Test case 3: Hook script exists but raises UndefinedError
    # Mock find_hook to return a script
    # Mock run_script_with_context to raise UndefinedError
    # Assert that UndefinedError is raised and project_dir is deleted if delete_project_on_failure is True
    
    # Test case 4: Hook script does not exist
    # Mock find_hook to return None
    # Assert that no exception is raised
    
    pass


# LLM-generated content at query #10
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Setup a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        script_path = tmpdir / "test_script.py"
        context = {"variable": "value"}
        script_content = "print('{{ variable }}')"
        script_path.write_text(script_content)

        run_script_with_context(script_path, tmpdir, context)

        # Verify the script was executed correctly
        # This assumes the script prints to stdout, which is not captured in the test
        # For a complete test, we'd need to capture stdout or verify side effects
        assert (tmpdir / "test_script.py").exists()



# LLM-generated content at query #11
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary script file
        script_path = Path(temp_dir) / "test_script.py"
        script_path.write_text("print('Hello, {{ name }}!')")

        # Define the context
        context = {"name": "World"}

        # Run the script with context
        run_script_with_context(script_path, temp_dir, context)

        # Assert that the script ran successfully
        assert True  # Placeholder assertion, replace with appropriate logic



# LLM-generated content at query #12
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Mock script path
    script_path = "mock_script.py"
    # Mock context
    context = {"key": "value"}
    # Mock cwd
    cwd = "/mock/cwd"
    # Mock run_script function
    run_script_mock = lambda path, cwd: None
    # Mock tempfile.NamedTemporaryFile
    tempfile.NamedTemporaryFile = lambda delete, mode, suffix: None
    # Mock Path.read_text
    Path.read_text = lambda path, encoding: "mock_content"
    # Mock jinja2.Environment.from_string
    create_env_with_context = lambda context: None
    env_from_string = lambda content: None
    template_render = lambda **kwargs: "mock_output"
    # Mock temp.write
    temp_write = lambda content: None
    # Run the function
    run_script_with_context(script_path, cwd, context)
    # Assertions
    assert True  # Placeholder assertion


# LLM-generated content at query #13
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test the run_hook_from_repo_dir function."""
    repo_dir = Path(__file__).parent.parent / "tests" / "fake-repo-pre"
    project_dir = Path(__file__).parent.parent / "tests" / "fake-repo-pre" / "output"
    context = {"cookiecutter": {"project_name": "test"}}
    delete_project_on_failure = True
    run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, delete_project_on_failure)


# LLM-generated content at query #14
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Mock the find_hook function to return a script path
    original_find_hook = find_hook
    find_hook = lambda hook_name, hooks_dir='hooks': ['/path/to/script.py']

    # Mock the run_script_with_context function
    original_run_script_with_context = run_script_with_context
    run_script_with_context = lambda script_path, cwd, context: None

    # Test with a valid hook name and context
    try:
        run_hook('pre_gen_project', '.', {'key': 'value'})
    except Exception as e:
        assert False, f"run_hook raised an exception: {e}"

    # Restore original functions
    find_hook = original_find_hook
    run_script_with_context = original_run_script_with_context

    # Test with no hooks found
    find_hook = lambda hook_name, hooks_dir='hooks': None
    run_hook('pre_gen_project', '.', {'key': 'value'})
    find_hook = original_find_hook

    print("All tests passed for run_hook")


# LLM-generated content at query #15
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy hook script
        hook_script = os.path.join(tmp_dir, 'hooks', 'pre_gen_project.py')
        os.makedirs(os.path.dirname(hook_script), exist_ok=True)
        with open(hook_script, 'w', encoding='utf-8') as f:
            f.write('print("Hello from pre_gen_project hook")')

        # Create a dummy project directory
        project_dir = os.path.join(tmp_dir, 'project')
        os.makedirs(project_dir)

        # Test running the hook
        run_hook('pre_gen_project', project_dir, {})

        # Verify the hook was executed (this is a simple test; in practice, you'd check side effects)
        assert os.path.exists(hook_script)


# LLM-generated content at query #16
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy hook script
        hook_script = os.path.join(tmp_dir, 'pre_gen_project.py')
        with open(hook_script, 'w', encoding='utf-8') as f:
            f.write('print("Hello from hook!")')
        
        # Make the script executable
        os.chmod(hook_script, 0o755)

        # Create a dummy project directory
        project_dir = os.path.join(tmp_dir, 'project')
        os.makedirs(project_dir)

        # Create a dummy context
        context = {'cookiecutter': {'project_name': 'test_project'}}

        # Test running the hook
        run_hook('pre_gen_project', project_dir, context)

        # Verify the hook was executed (this is a simple test; in a real scenario, you might check for side effects)
        assert True  # Placeholder assertion; replace with actual checks if needed


# LLM-generated content at query #17
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Setup: Create a temporary directory with a pre_prompt hook script
    with tempfile.TemporaryDirectory() as repo_dir:
        hooks_dir = Path(repo_dir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.py'
        script_path.write_text('print("Running pre_prompt hook")')

        # Execute: Call the function
        result_dir = run_pre_prompt_hook(repo_dir)

        # Verify: Ensure the function returns the correct directory
        assert result_dir == repo_dir

        # Verify: Ensure the hook script was executed (you can add more sophisticated checks)
        # For example, you could capture stdout and check for the printed message

    # Test failure case: Invalid script
    with tempfile.TemporaryDirectory() as repo_dir:
        hooks_dir = Path(repo_dir) / 'hooks'
        hooks_dir.mkdir()
        script_path = hooks_dir / 'pre_prompt.py'
        script_path.write_text('raise Exception("Failed hook")')

        try:
            run_pre_prompt_hook(repo_dir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException"


# LLM-generated content at query #18
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Mock repository directory
    repo_dir = Path(tempfile.mkdtemp())
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir(exist_ok=True)
    pre_prompt_script = hooks_dir / 'pre_prompt.py'
    pre_prompt_script.write_text('import sys\nsys.exit(0)')

    # Test successful execution
    try:
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
    finally:
        rmtree(repo_dir)

    # Test failure execution
    repo_dir = Path(tempfile.mkdtemp())
    hooks_dir = repo_dir / 'hooks'
    hooks_dir.mkdir(exist_ok=True)
    pre_prompt_script = hooks_dir / 'pre_prompt.py'
    pre_prompt_script.write_text('import sys\nsys.exit(1)')

    try:
        run_pre_prompt_hook(repo_dir)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        pass
    finally:
        rmtree(repo_dir)

    # Test no hooks directory
    repo_dir = Path(tempfile.mkdtemp())
    try:
        result = run_pre_prompt_hook(repo_dir)
        assert result == repo_dir
    finally:
        rmtree(repo_dir)


# LLM-generated content at query #19
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test run_script_with_context function."""
    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp:
        temp.write('print("Hello, World!")')
        temp_path = temp.name

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        context = {'project_name': 'TestProject'}

        # Test successful execution
        try:
            run_script_with_context(temp_path, temp_dir, context)
        except FailedHookException:
            assert False, "run_script_with_context should not raise an exception"

    # Clean up
    os.unlink(temp_path)


# LLM-generated content at query #20
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test run_script_with_context function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary script file
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, {{ name }}!")')

        # Define context
        context = {'name': 'World'}

        # Run the script with context
        run_script_with_context(script_path, temp_dir, context)

        # Check if the script executed successfully
        assert True  # If no exception is raised, the test passes


# LLM-generated content at query #21
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert find_hook('pre_gen_project', tmp_dir) is None

    # Test case 3: Valid hook file
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_file = os.path.join(tmp_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', tmp_dir) == [os.path.abspath(hook_file)]

    # Test case 4: Backup file (should be ignored)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_file = os.path.join(tmp_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        assert find_hook('pre_gen_project', tmp_dir) is None

    # Test case 5: Unsupported hook (should be ignored)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_file = os.path.join(tmp_dir, 'unsupported_hook.py')
        Path(hook_file).touch()
        assert find_hook('unsupported_hook', tmp_dir) is None

    # Test case 6: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_file1 = os.path.join(tmp_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(tmp_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', tmp_dir)
        assert sorted(result) == sorted([os.path.abspath(hook_file1), os.path.abspath(hook_file2)])


# LLM-generated content at query #22
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Step 1: Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Step 2: Create a script file with Jinja templating
    script_path = os.path.join(temp_dir, 'test_script.py')
    script_content = """
import os
print("Template variable: {{ my_var }}")
    """
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Step 3: Define a context dictionary
    context = {'my_var': 'Hello, World!'}
    
    # Step 4: Run the script with the context
    run_script_with_context(script_path, temp_dir, context)
    
    # Step 5: Clean up the temporary directory
    os.remove(os.path.join(temp_dir, 'test_script.py'))
    os.rmdir(temp_dir)


# LLM-generated content at query #23
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test cases
    # Case 1: No hooks directory
    assert find_hook('pre_gen_project') is None

    # Case 2: Valid hooks directory with valid hook
    hooks_dir = tempfile.mkdtemp()
    hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
    with open(hook_file, 'w') as f:
        f.write('print("Hello World")')
    assert find_hook('pre_gen_project', hooks_dir) == [hook_file]

    # Case 3: Valid hooks directory with invalid hook
    hook_file = os.path.join(hooks_dir, 'invalid_hook.py')
    with open(hook_file, 'w') as f:
        f.write('print("Hello World")')
    assert find_hook('pre_gen_project', hooks_dir) == [os.path.join(hooks_dir, 'pre_gen_project.py')]

    # Clean up
    os.remove(hook_file)
    os.rmdir(hooks_dir)



# LLM-generated content at query #24
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Setup: Create a temporary script with Jinja2 placeholders
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp:
        temp.write("""{{ project_name }}""")
        temp_path = temp.name

    # Define context
    context = {'project_name': 'TestProject'}

    # Run the function
    run_script_with_context(temp_path, '.', context)

    # Verify: Check if the script executed successfully
    assert True  # Replace with actual verification logic

    # Cleanup
    os.remove(temp_path)


# LLM-generated content at query #25
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    """Test the run_pre_prompt_hook function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock hook script
        hook_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hook_dir)
        hook_script = os.path.join(hook_dir, 'pre_prompt.py')
        with open(hook_script, 'w') as f:
            f.write('#!/usr/bin/env python\nprint("Pre-prompt hook")')

        # Test with valid hook
        result = run_pre_prompt_hook(tmpdir)
        assert os.path.exists(result)

        # Test with invalid hook directory
        invalid_dir = os.path.join(tmpdir, 'nonexistent')
        result = run_pre_prompt_hook(invalid_dir)
        assert result == invalid_dir

        # Test with failing hook script
        failing_script = os.path.join(hook_dir, 'failing_pre_prompt.py')
        with open(failing_script, 'w') as f:
            f.write('#!/usr/bin/env python\nraise Exception("Hook failed")')
        try:
            run_pre_prompt_hook(tmpdir)
        except FailedHookException:
            pass
        else:
            assert False, "Expected FailedHookException"

if __name__ == '__main__':
    test_run_pre_prompt_hook()


# LLM-generated content at query #26
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Mocking subprocess.Popen to simulate script execution
    # Ensure correct script execution and error handling
    pass


# LLM-generated content at query #27
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir function."""
    # Setup test environment
    test_repo_dir = Path(tempfile.mkdtemp())
    test_project_dir = Path(tempfile.mkdtemp())
    test_context = {'project_name': 'test_project'}

    # Create a test hook script
    hook_script = test_repo_dir / 'hooks' / 'pre_gen_project.sh'
    hook_script.parent.mkdir()
    hook_script.write_text('#!/bin/bash\necho "Running pre_gen_project hook"')

    # Test successful hook execution
    try:
        run_hook_from_repo_dir(
            test_repo_dir,
            'pre_gen_project',
            test_project_dir,
            test_context,
            True,
        )
    except FailedHookException:
        assert False, "Hook execution failed unexpectedly"

    # Test hook failure with delete_project_on_failure=True
    hook_script.write_text('#!/bin/bash\nexit 1')
    try:
        run_hook_from_repo_dir(
            test_repo_dir,
            'pre_gen_project',
            test_project_dir,
            test_context,
            True,
        )
        assert False, "Expected FailedHookException not raised"
    except FailedHookException:
        assert not test_project_dir.exists(), "Project directory not deleted on failure"

    # Cleanup
    rmtree(test_repo_dir)


# LLM-generated content at query #28
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    pass


# LLM-generated content at query #29
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory exists
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Hooks directory exists but no matching hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 3: Hooks directory exists with matching hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello")')
        assert len(find_hook('pre_gen_project', hooks_dir)) == 1

    # Test case 4: Multiple matching hooks
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        with open(hook_file1, 'w') as f:
            f.write('print("Hello")')
        with open(hook_file2, 'w') as f:
            f.write('echo "Hello"')
        assert len(find_hook('pre_gen_project', hooks_dir)) == 2

    # Test case 5: Backup file should be ignored
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('print("Hello")')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test case 6: Unsupported hook should be ignored
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello")')
        assert find_hook('unsupported_hook', hooks_dir) is None


# LLM-generated content at query #30
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test run_hook_from_repo_dir function."""
    repo_dir = Path('tests/test-repo')
    hook_name = 'pre_gen_project'
    project_dir = Path('tests/test-project')
    context = {'project_name': 'test-project'}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass


# LLM-generated content at query #31
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Test with a valid script path, working directory, and context
    script_path = "test_script.py"
    cwd = "."
    context = {"key": "value"}
    try:
        run_script_with_context(script_path, cwd, context)
    except FailedHookException as e:
        assert str(e) == "Hook script failed (error: [Errno 2] No such file or directory: 'test_script.py')"

    # Test with an invalid script path
    script_path = "invalid_script.py"
    try:
        run_script_with_context(script_path, cwd, context)
    except FailedHookException as e:
        assert str(e) == "Hook script failed (error: [Errno 2] No such file or directory: 'invalid_script.py')"

    # Test with a valid script path and working directory but empty context
    script_path = "test_script.py"
    context = {}
    try:
        run_script_with_context(script_path, cwd, context)
    except FailedHookException as e:
        assert str(e) == "Hook script failed (error: [Errno 2] No such file or directory: 'test_script.py')"


# LLM-generated content at query #32
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Test case 1: Successful hook execution
    repo_dir = Path("test_repo")
    hook_name = "test_hook"
    project_dir = Path("test_project")
    context = {"key": "value"}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert True
    except FailedHookException:
        assert False

    # Test case 2: Failed hook execution with project directory deletion
    repo_dir = Path("test_repo_fail")
    hook_name = "test_hook_fail"
    project_dir = Path("test_project_fail")
    context = {"key": "value"}
    delete_project_on_failure = True
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except FailedHookException:
        assert not project_dir.exists()

    # Test case 3: Failed hook execution without project directory deletion
    repo_dir = Path("test_repo_fail_no_delete")
    hook_name = "test_hook_fail_no_delete"
    project_dir = Path("test_project_fail_no_delete")
    context = {"key": "value"}
    delete_project_on_failure = False
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False
    except FailedHookException:
        assert project_dir.exists()


# LLM-generated content at query #33
#--------------------------

# Unit test for function run_script
def test_run_script():
    # Create a temporary directory and a simple script
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write('print("Hello, World!")')
        
        # Test running the script
        run_script(script_path, tmp_dir)
        
        # Test a failing script
        failing_script_path = os.path.join(tmp_dir, 'fail_script.py')
        with open(failing_script_path, 'w') as f:
            f.write('import sys; sys.exit(1)')
        
        try:
            run_script(failing_script_path, tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass
        
        # Test a non-executable script
        non_exec_script_path = os.path.join(tmp_dir, 'non_exec_script.py')
        with open(non_exec_script_path, 'w') as f:
            f.write('print("Hello, World!")')
        os.chmod(non_exec_script_path, 0o644)  # Remove executable permissions
        
        try:
            run_script(non_exec_script_path, tmp_dir)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass


# LLM-generated content at query #34
#--------------------------

# Unit test for function run_script
def test_run_script():
    """Test the run_script function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script file
        script_path = os.path.join(tmp_dir, 'test_script.sh')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Run the script
        run_script(script_path, tmp_dir)
        
        # Verify the script ran successfully (no exception raised)
        assert True


# LLM-generated content at query #35
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Setup test environment
    repo_dir = Path("test_repo")
    hook_name = "pre_gen_project"
    project_dir = Path("test_project")
    context = {"project_name": "Test Project"}
    delete_project_on_failure = True

    # Create test files
    repo_dir.mkdir(exist_ok=True)
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_script = hooks_dir / "pre_gen_project.py"
    hook_script.write_text("print('Running pre_gen_project hook')")

    # Test successful hook execution
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    # Test failed hook execution
    hook_script.write_text("import sys\nsys.exit(1)")
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert not project_dir.exists(), "Project directory should be deleted on hook failure"

    # Cleanup test environment
    rmtree(repo_dir)


# LLM-generated content at query #36
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'key': 'value'}
    hook_name = 'pre_gen_project'
    delete_project_on_failure = True

    # Create a dummy hook script
    hook_script = os.path.join(repo_dir, 'hooks', hook_name + '.py')
    os.makedirs(os.path.dirname(hook_script))
    with open(hook_script, 'w') as f:
        f.write('print("Hello from hook")')

    # Test successful execution
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except Exception as e:
        assert False, f"Unexpected exception: {e}"

    # Test failure with delete_project_on_failure=True
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert False, "Expected FailedHookException"
    except FailedHookException:
        assert not os.path.exists(project_dir), "Project directory should be deleted on failure"

    # Cleanup
    rmtree(repo_dir)


# LLM-generated content at query #37
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    # Setup test environment
    repo_dir = Path(tempfile.mkdtemp())
    project_dir = Path(tempfile.mkdtemp())
    context = {"project_name": "test_project"}
    delete_project_on_failure = True

    # Create a dummy hook script
    hook_name = "pre_gen_project"
    hooks_dir = repo_dir / "hooks"
    hooks_dir.mkdir()
    script_path = hooks_dir / f"{hook_name}.py"
    script_path.write_text("print('Running pre_gen_project hook')")

    # Run the hook
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

    # Cleanup
    rmtree(repo_dir)
    rmtree(project_dir)


# LLM-generated content at query #38
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test the run_hook_from_repo_dir function."""
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'key': 'value'}
    delete_project_on_failure = True

    # Test with no hook
    run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, delete_project_on_failure)

    # Test with a failing hook
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.join(repo_dir, 'hooks'))
    with open(hook_script, 'w') as f:
        f.write('import sys\nsys.exit(1)')

    try:
        run_hook_from_repo_dir(repo_dir, 'pre_gen_project', project_dir, context, delete_project_on_failure)
    except FailedHookException:
        pass  # Expected
    else:
        assert False, "Expected FailedHookException"

    # Cleanup
    rmtree(repo_dir)
    rmtree(project_dir)


# LLM-generated content at query #39
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary script file
        script_path = Path(temp_dir) / "test_script.py"
        script_content = "print('Hello, {{ name }}!')"
        script_path.write_text(script_content)

        # Define the context
        context = {"name": "World"}

        # Run the script with context
        run_script_with_context(script_path, temp_dir, context)

        # Check if the script executed successfully
        # Since the script prints to stdout, we can't capture the output directly
        # So we assume that if no exception is raised, the script executed successfully
        assert True


# LLM-generated content at query #40
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test case 1: No hooks directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test case 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert find_hook('pre_gen_project', tmp_dir) is None

    # Test case 3: Valid hook file
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.mkdir(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('print("Hello World")')
        assert find_hook('pre_gen_project', hook_dir) == [hook_file]

    # Test case 4: Invalid hook file (backup file)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.mkdir(hook_dir)
        hook_file = os.path.join(hook_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('print("Hello World")')
        assert find_hook('pre_gen_project', hook_dir) is None

    # Test case 5: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmp_dir:
        hook_dir = os.path.join(tmp_dir, 'hooks')
        os.mkdir(hook_dir)
        hook_file1 = os.path.join(hook_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hook_dir, 'post_gen_project.py')
        with open(hook_file1, 'w') as f:
            f.write('print("Hello World")')
        with open(hook_file2, 'w') as f:
            f.write('print("Hello World")')
        assert find_hook('pre_gen_project', hook_dir) == [hook_file1]
        assert find_hook('post_gen_project', hook_dir) == [hook_file2]


# LLM-generated content at query #41
#--------------------------

# Unit test for function run_pre_prompt_hook
def test_run_pre_prompt_hook():
    # Mock a repository directory and a pre_prompt hook script
    repo_dir = tempfile.mkdtemp()
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    script_path = os.path.join(hooks_dir, 'pre_prompt.py')
    with open(script_path, 'w') as f:
        f.write('print("Pre-prompt hook executed")')

    # Test that the hook runs successfully
    assert run_pre_prompt_hook(repo_dir) == repo_dir

    # Modify the script to simulate a failure
    with open(script_path, 'w') as f:
        f.write('import sys\nsys.exit(1)')

    # Test that the hook failure raises an exception
    try:
        run_pre_prompt_hook(repo_dir)
    except FailedHookException:
        pass
    else:
        assert False, "FailedHookException not raised"

    # Clean up
    rmtree(repo_dir)


# LLM-generated content at query #42
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test the run_hook_from_repo_dir function."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmp_dir) / "project"
        project_dir.mkdir()
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "pre_gen_project.py"
        hook_file.write_text("print('Hello, world!')")
        context = {}

        # Test successful hook execution
        run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, True)
        assert project_dir.exists(), "Project directory should exist after successful hook"

        # Test failed hook execution
        hook_file.write_text("import sys; sys.exit(1)")
        try:
            run_hook_from_repo_dir(repo_dir, "pre_gen_project", project_dir, context, True)
        except FailedHookException:
            assert not project_dir.exists(), "Project directory should be deleted on hook failure"
        else:
            assert False, "FailedHookException should be raised on hook failure"


# LLM-generated content at query #43
#--------------------------

# Unit test for function run_script
def test_run_script():
    """Test the run_script function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple script
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, world!")')
        
        # Run the script
        run_script(script_path, temp_dir)
        
        # Check if the script was executed successfully
        # (no exception should be raised)
        assert True


# LLM-generated content at query #44
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    # Test1: Hook directory is missing
    assert find_hook('pre_prompt', 'missing_hooks_dir') is None

    # Test2: Provide hook directory with hook files
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, 'hooks'))
        # Add a valid hook file
        with open(os.path.join(tmp_dir, 'hooks', 'pre_prompt.py'), 'w') as f:
            f.write('')
        # Add a invalid hook file
        with open(os.path.join(tmp_dir, 'hooks', 'invalid_hook.py'), 'w') as f:
            f.write('')
        # Add a backup hook file
        with open(os.path.join(tmp_dir, 'hooks', 'pre_prompt.py~'), 'w') as f:
            f.write('')
        # Add a valid hook file
        with open(os.path.join(tmp_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('')
        # Test with valid hook
        assert len(find_hook('pre_prompt', os.path.join(tmp_dir, 'hooks'))) == 1
        # Test with invalid hook
        assert find_hook('invalid_hook', os.path.join(tmp_dir, 'hooks')) is None
        # Test with backup hook
        assert find_hook('pre_prompt.py~', os.path.join(tmp_dir, 'hooks')) is None
        # Test with valid hook
        assert len(find_hook('pre_gen_project', os.path.join(tmp_dir, 'hooks'))) == 1



# LLM-generated content at query #45
#--------------------------

# Unit test for function find_hook
def test_find_hook():
    """Test the find_hook function."""
    # Test with a non-existent hook directory
    assert find_hook('pre_gen_project', 'nonexistent_dir') is None

    # Test with an empty hooks directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with a valid hook script
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        with open(hook_file, 'w') as f:
            f.write('')
        assert find_hook('pre_gen_project', hooks_dir) == [os.path.abspath(hook_file)]

    # Test with a backup file
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        with open(hook_file, 'w') as f:
            f.write('')
        assert find_hook('pre_gen_project', hooks_dir) is None

    # Test with an unsupported hook
    with tempfile.TemporaryDirectory() as tmp_dir:
        hooks_dir = os.path.join(tmp_dir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        with open(hook_file, 'w') as f:
            f.write('')
        assert find_hook('unsupported_hook', hooks_dir) is None


# LLM-generated content at query #46
#--------------------------

# Unit test for function run_hook
def test_run_hook():
    """Test the run_hook function."""
    # Setup
    hook_name = "pre_gen_project"
    project_dir = Path("test_project")
    context = {"project_name": "test_project"}

    # Test with no scripts found
    assert run_hook(hook_name, project_dir, context) is None

    # Test with a valid script
    # Create a temporary hook script
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
        temp.write(b"print('Hello, world!')")
        temp_path = temp.name

    # Mock find_hook to return the temporary script
    original_find_hook = find_hook
    find_hook = lambda hook_name, hooks_dir="hooks": [temp_path]

    try:
        run_hook(hook_name, project_dir, context)
    finally:
        # Cleanup
        os.unlink(temp_path)
        find_hook = original_find_hook

    # Test with a failing script
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
        temp.write(b"import sys; sys.exit(1)")
        temp_path = temp.name

    find_hook = lambda hook_name, hooks_dir="hooks": [temp_path]

    try:
        try:
            run_hook(hook_name, project_dir, context)
            assert False, "Expected FailedHookException"
        except FailedHookException:
            pass
    finally:
        os.unlink(temp_path)
        find_hook = original_find_hook


# LLM-generated content at query #47
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test script file
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('print("Hello, {{ name }}!")')

        # Define context
        context = {'name': 'World'}

        # Run the script with context
        run_script_with_context(script_path, temp_dir, context)

        # Check if the script executed successfully
        # (In a real test, you'd capture stdout or check side effects)
        assert True  # Placeholder assertion


# LLM-generated content at query #48
#--------------------------

# Unit test for function run_script
def test_run_script():
    """Test the run_script function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple script file
        script_path = os.path.join(tmp_dir, 'test_script.sh')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('#!/bin/sh\necho "Hello, World!"\n')
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Test running the script
        try:
            run_script(script_path, tmp_dir)
        except FailedHookException as e:
            assert False, f"run_script failed: {e}"
        
        # Test running a non-existent script
        non_existent_script = os.path.join(tmp_dir, 'non_existent.sh')
        try:
            run_script(non_existent_script, tmp_dir)
            assert False, "Expected FailedHookException for non-existent script"
        except FailedHookException:
            pass
        
        # Test running an empty script
        empty_script = os.path.join(tmp_dir, 'empty_script.sh')
        with open(empty_script, 'w', encoding='utf-8') as f:
            f.write('')
        os.chmod(empty_script, 0o755)
        try:
            run_script(empty_script, tmp_dir)
            assert False, "Expected FailedHookException for empty script"
        except FailedHookException:
            pass


# LLM-generated content at query #49
#--------------------------

# Unit test for function run_hook_from_repo_dir
def test_run_hook_from_repo_dir():
    """Test the run_hook_from_repo_dir function."""
    # Setup
    repo_dir = tempfile.mkdtemp()
    project_dir = tempfile.mkdtemp()
    context = {'key': 'value'}
    delete_project_on_failure = True

    # Test case 1: No hook script found
    run_hook_from_repo_dir(repo_dir, 'nonexistent_hook', project_dir, context, delete_project_on_failure)
    assert os.path.exists(project_dir), "Project directory should not be deleted when no hook is found"

    # Test case 2: Hook script found and runs successfully
    hook_name = 'pre_gen_project'
    script_path = os.path.join(repo_dir, 'hooks', f'{hook_name}.py')
    os.makedirs(os.path.join(repo_dir, 'hooks'))
    with open(script_path, 'w') as f:
        f.write('print("Hello, world!")')
    run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert os.path.exists(project_dir), "Project directory should not be deleted when hook runs successfully"

    # Test case 3: Hook script fails and delete_project_on_failure is True
    with open(script_path, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    except FailedHookException:
        assert not os.path.exists(project_dir), "Project directory should be deleted when hook fails and delete_project_on_failure is True"

    # Cleanup
    rmtree(repo_dir)
    if os.path.exists(project_dir):
        rmtree(project_dir)


# LLM-generated content at query #50
#--------------------------

# Unit test for function run_script_with_context
def test_run_script_with_context():
    """Test run_script_with_context function."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp:
        temp.write('print("Hello, World!")')
        temp.flush()
        context = {'project_name': 'test_project'}
        run_script_with_context(temp.name, '.', context)


