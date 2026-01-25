####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function apply_overwrites_to_context
def test_apply_overwrites_to_context():
    context = {
        "cookiecutter": {
            "choice_var": ["a", "b", "c"],
            "multi_choice_var": ["x", "y", "z"],
            "bool_var": True,
            "dict_var": {"key1": "value1", "key2": "value2"},
        }
    }
    overwrite_context = {
        "choice_var": "b",
        "multi_choice_var": ["y", "z"],
        "bool_var": "no",
        "dict_var": {"key2": "new_value2", "key3": "value3"},
    }
    apply_overwrites_to_context(context["cookiecutter"], overwrite_context)
    assert context["cookiecutter"]["choice_var"] == ["b", "a", "c"]
    assert context["cookiecutter"]["multi_choice_var"] == ["y", "z"]
    assert context["cookiecutter"]["bool_var"] is False
    assert context["cookiecutter"]["dict_var"] == {
        "key1": "value1",
        "key2": "new_value2",
        "key3": "value3",
    }

    # Test invalid overwrite for choice variable
    try:
        apply_overwrites_to_context(context["cookiecutter"], {"choice_var": "d"})
    except ValueError as e:
        assert "d provided for choice variable choice_var" in str(e)

    # Test invalid overwrite for multi-choice variable
    try:
        apply_overwrites_to_context(context["cookiecutter"], {"multi_choice_var": ["w"]})
    except ValueError as e:
        assert "['w'] provided for multi-choice variable multi_choice_var" in str(e)

    # Test invalid overwrite for boolean variable
    try:
        apply_overwrites_to_context(context["cookiecutter"], {"bool_var": "maybe"})
    except ValueError as e:
        assert "maybe provided for variable bool_var" in str(e)


# LLM-generated content at query #2
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    # Test case 1: Normal directory name
    dirname = "test_dir"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    output_dir = Path("/tmp")
    environment = Environment()
    path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert path == Path("/tmp/test_dir")
    assert created == True
    os.rmdir(path)

    # Test case 2: Empty directory name
    dirname = ""
    context = {"cookiecutter": {"project_name": "TestProject"}}
    output_dir = Path("/tmp")
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException"

    # Test case 3: Directory already exists and overwrite_if_exists is False
    dirname = "existing_dir"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    output_dir = Path("/tmp")
    environment = Environment()
    Path("/tmp/existing_dir").mkdir()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        pass
    else:
        assert False, "Expected OutputDirExistsException"
    os.rmdir(Path("/tmp/existing_dir"))

    # Test case 4: Directory already exists and overwrite_if_exists is True
    dirname = "existing_dir"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    output_dir = Path("/tmp")
    environment = Environment()
    Path("/tmp/existing_dir").mkdir()
    path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert path == Path("/tmp/existing_dir")
    assert created == False
    os.rmdir(path)


# LLM-generated content at query #3
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    """Test the render_and_create_dir function."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = Path('/tmp')
    environment = Environment(loader=FileSystemLoader('/tmp'))
    dirname = '{{ cookiecutter.project_name }}'
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_path == Path('/tmp/test_project')
    assert created is True
    assert dir_path.exists()
    dir_path.rmdir()


# LLM-generated content at query #4
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    project_dir = "test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"_copy_without_render": [], "_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))
    
    # Create a test template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ name }}!")
    
    # Test rendering
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    
    # Verify the output file
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello World!"
    
    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #5
#--------------------------

# Unit test for function apply_overwrites_to_context
def test_apply_overwrites_to_context():
    context = {'key1': 'value1', 'key2': 'value2'}
    overwrite_context = {'key1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['key1'] == 'new_value1'
    assert context['key2'] == 'value2'



# LLM-generated content at query #6
#--------------------------

# Unit test for function is_copy_only_path
def test_is_copy_only_path():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'images/*']
        }
    }
    assert is_copy_only_path('file.txt', context) == True
    assert is_copy_only_path('images/photo.jpg', context) == True
    assert is_copy_only_path('scripts/script.py', context) == False
    assert is_copy_only_path('README.md', context) == False



# LLM-generated content at query #7
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    project_dir = "test_project"
    infile = "test_file.txt"
    context = {"cookiecutter": {"_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))
    
    # Create a test file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello, {{ name }}!")
    
    # Test rendering
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    
    # Verify the output
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello, World!"
    
    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #8
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    repo_dir = Path(temp_dir) / "template"
    repo_dir.mkdir()
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir()

    # Create a simple template
    template_file = repo_dir / "{{cookiecutter.project_name}}.txt"
    template_file.write_text("Hello, {{cookiecutter.project_name}}!")

    # Create a cookiecutter.json file
    context_file = repo_dir / "cookiecutter.json"
    context_file.write_text('{"project_name": "TestProject"}')

    # Test
    try:
        project_dir = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True,
        )
        assert Path(project_dir).exists()
        generated_file = Path(project_dir) / "TestProject.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Hello, TestProject!"
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


# LLM-generated content at query #9
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    project_dir = 'test_project_dir'
    infile = 'test_infile.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))

    # Create a temporary test file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter._new_lines }}')

    generate_file(project_dir, infile, context, env)

    # Verify the output file content
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'r', encoding='utf-8') as f:
        assert f.read() == 'Hello, \n'

    # Clean up
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #10
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Create a temporary directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test template file
        template_file = os.path.join(tmp_dir, 'test_template.txt')
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write("Hello {{ name }}!")

        # Create a test context
        context = {'name': 'World'}

        # Create a Jinja2 environment
        env = Environment(loader=FileSystemLoader(tmp_dir))

        # Generate the file
        output_dir = os.path.join(tmp_dir, 'output')
        os.makedirs(output_dir)
        generate_file(output_dir, 'test_template.txt', context, env)

        # Check the generated file
        generated_file = os.path.join(output_dir, 'test_template.txt')
        assert os.path.exists(generated_file)
        with open(generated_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == "Hello World!"

        # Test binary file copy
        binary_file = os.path.join(tmp_dir, 'test_binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')
        generate_file(output_dir, 'test_binary.bin', context, env)
        generated_binary = os.path.join(output_dir, 'test_binary.bin')
        assert os.path.exists(generated_binary)
        with open(generated_binary, 'rb') as f:
            content = f.read()
            assert content == b'\x00\x01\x02\x03'

        # Test skip_if_file_exists
        generate_file(output_dir, 'test_template.txt', context, env, skip_if_file_exists=True)
        # Should not have modified the existing file
        with open(generated_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == "Hello World!"


# LLM-generated content at query #11
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Set up test environment
    test_project_dir = "test_project"
    test_infile = "test_template.txt"
    test_context = {"name": "Test Project"}
    test_env = Environment(loader=FileSystemLoader("."))
    
    # Create a temporary template file
    with open(test_infile, "w") as file:
        file.write("Project: {{ name }}")
    
    # Generate the file
    generate_file(test_project_dir, test_infile, test_context, test_env)
    
    # Verify the generated file
    expected_outfile = os.path.join(test_project_dir, test_infile)
    with open(expected_outfile, "r") as file:
        content = file.read()
        assert content == "Project: Test Project"
    
    # Clean up
    os.remove(test_infile)
    shutil.rmtree(test_project_dir)



# LLM-generated content at query #12
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    # Setup
    repo_dir = Path("tests/fake-repo")
    context = {"cookiecutter": {"project_name": "Test Project"}}
    output_dir = Path("tests/output")
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    # Test case 1: Successful generation
    try:
        project_dir = generate_files(
            repo_dir,
            context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure,
        )
        assert Path(project_dir).exists()
    finally:
        if Path(project_dir).exists():
            shutil.rmtree(project_dir)

    # Test case 2: Undefined variable in template
    context_fail = {"cookiecutter": {"undefined_var": "{{ undefined_var }}"}}
    try:
        generate_files(
            repo_dir,
            context_fail,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure,
        )
        assert False, "Expected UndefinedVariableInTemplate exception"
    except UndefinedVariableInTemplate:
        pass
    except Exception as e:
        assert False, f"Unexpected exception: {e}"

    # Test case 3: Output directory exists and overwrite_if_exists is False
    overwrite_if_exists = False
    try:
        # Create the output directory first
        os.makedirs(output_dir / "fake-repo", exist_ok=True)
        generate_files(
            repo_dir,
            context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure,
        )
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass
    finally:
        if Path(output_dir / "fake-repo").exists():
            shutil.rmtree(output_dir / "fake-repo")

    print("All tests passed!")

if __name__ == "__main__":
    test_generate_files()


# LLM-generated content at query #13
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Create a temporary directory and files for testing
    import tempfile
    from pathlib import Path
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        # Setup test environment
        project_dir = Path(temp_dir) / "project"
        project_dir.mkdir()
        template_dir = Path(temp_dir) / "template"
        template_dir.mkdir()

        # Create a test template file
        test_file = template_dir / "test.txt"
        test_file.write_text("Hello {{ name }}!")

        # Create context and environment
        context = {'name': 'World'}
        env = Environment(loader=FileSystemLoader(template_dir))

        # Test generating a file
        generate_file(str(project_dir), str(test_file.name), context, env)

        # Verify the generated file
        generated_file = project_dir / "test.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Hello World!"

    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #14
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    repo_dir = Path("tests/fake-repo-tmpl")
    context = {
        "cookiecutter": {
            "project_name": "Test Project",
            "project_slug": "test_project",
            "author_name": "Test Author",
            "email": "test@example.com",
            "version": "0.1.0",
            "license": "MIT",
            "_copy_without_render": ["*.txt"],
        }
    }
    output_dir = Path("tests/fake-repo")
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    assert os.path.exists(project_dir)
    assert os.path.exists(os.path.join(project_dir, "README.md"))
    assert os.path.exists(os.path.join(project_dir, "LICENSE"))
    assert os.path.exists(os.path.join(project_dir, "requirements.txt"))

    shutil.rmtree(project_dir)


# LLM-generated content at query #15
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Create a temporary context file
    context_file = 'test_cookiecutter.json'
    default_context = {'name': 'default_name'}
    extra_context = {'name': 'extra_name'}

    # Write the temporary context file
    with open(context_file, 'w') as f:
        json.dump({'name': 'test_name'}, f)

    # Test with default_context only
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['name'] == 'test_name'

    # Test with extra_context
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['name'] == 'extra_name'

    # Clean up the temporary context file
    os.remove(context_file)


# LLM-generated content at query #16
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Create a temporary directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test template file
        template_path = os.path.join(tmp_dir, 'template.txt')
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write("Hello, {{ name }}!")

        # Create a test project directory
        project_dir = os.path.join(tmp_dir, 'project')
        os.makedirs(project_dir)

        # Define the context
        context = {'name': 'Cookiecutter'}

        # Create a Jinja2 environment
        env = Environment(loader=FileSystemLoader(tmp_dir))

        # Generate the file
        generate_file(project_dir, 'template.txt', context, env)

        # Verify the generated file
        generated_file_path = os.path.join(project_dir, 'template.txt')
        assert os.path.exists(generated_file_path)
        with open(generated_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == "Hello, Cookiecutter!"



# LLM-generated content at query #17
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_file.txt"
    test_file.write_text("Hello {{ name }}!", encoding="utf-8")
    
    context = {"cookiecutter": {"name": "World", "_new_lines": False}}
    env = Environment(loader=FileSystemLoader(test_dir))
    
    output_dir = Path("output_dir")
    output_dir.mkdir(exist_ok=True)
    
    # Test
    generate_file(str(output_dir), str(test_file), context, env)
    
    # Verify
    output_file = output_dir / "test_file.txt"
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "Hello World!"
    
    # Cleanup
    test_file.unlink()
    output_file.unlink()
    test_dir.rmdir()
    output_dir.rmdir()


# LLM-generated content at query #18
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    # Setup test environment
    repo_dir = Path("tests/fake-repo")
    context = generate_context(repo_dir / "cookiecutter.json")
    output_dir = Path("tests/output")

    # Clean up previous test outputs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Test generation with overwrite_if_exists
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    # Assertions
    assert Path(project_dir).exists()
    assert (Path(project_dir) / "README.rst").exists()

    # Test generation with skip_if_file_exists
    project_dir_skip = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(project_dir_skip).exists()
    assert (Path(project_dir_skip) / "README.rst").exists()

    # Clean up
    shutil.rmtree(output_dir)


# LLM-generated content at query #19
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    """Test the render_and_create_dir function."""
    # Create a temporary directory for testing
    import tempfile
    tmp_dir = tempfile.mkdtemp()

    # Test case 1: Valid directory name
    dirname = "test_dir"
    context = {}
    output_dir = tmp_dir
    environment = Environment()
    result_dir, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert os.path.exists(result_dir)
    assert created

    # Test case 2: Empty directory name
    dirname = ""
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

    # Test case 3: Directory already exists, overwrite_if_exists=True
    dirname = "existing_dir"
    os.mkdir(os.path.join(tmp_dir, dirname))
    result_dir, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert os.path.exists(result_dir)
    assert not created

    # Test case 4: Directory already exists, overwrite_if_exists=False
    dirname = "another_existing_dir"
    os.mkdir(os.path.join(tmp_dir, dirname))
    try:
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

    # Clean up
    shutil.rmtree(tmp_dir)


# LLM-generated content at query #20
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid JSON file
    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context('test.json')
    assert context == {'test': {'key': 'value'}}
    os.remove('test.json')

    # Test with an invalid JSON file
    with open('test.json', 'w', encoding='utf-8') as f:
        f.write('invalid json')
    try:
        generate_context('test.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Should have raised ContextDecodingException"
    os.remove('test.json')

    # Test with default_context
    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context('test.json', default_context={'key': 'default'})
    assert context == {'test': {'key': 'value'}}
    os.remove('test.json')

    # Test with extra_context
    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context('test.json', extra_context={'key': 'extra'})
    assert context == {'test': {'key': 'extra'}}
    os.remove('test.json')


# LLM-generated content at query #21
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Setup
    project_dir = '/tmp/project_dir'
    infile = 'template.txt'
    context = {'variable': 'value'}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    # Create a template file
    with open(infile, 'w') as f:
        f.write('{{ variable }}')

    # Execute
    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # Verify
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'value'

    # Cleanup
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #22
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid JSON file
    context_file = 'tests/test-context/cookiecutter.json'
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert 'project_name' in context['cookiecutter']

    # Test with a non-existent JSON file
    try:
        generate_context('non-existent.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with an invalid JSON file
    try:
        generate_context('tests/test-context/invalid.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with default_context
    default_context = {'cookiecutter': {'project_name': 'Default Project'}}
    context = generate_context(context_file, default_context=default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'cookiecutter': {'project_name': 'Extra Project'}}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'


# LLM-generated content at query #23
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    # Setup test data
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    # Mock necessary functions
    def mock_find_template(repo_dir, env):
        return "test_template_dir"

    def mock_run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure):
        pass

    def mock_work_in(template_dir):
        yield

    def mock_is_copy_only_path(path, context):
        return False

    def mock_generate_file(project_dir, infile, context, env, skip_if_file_exists):
        pass

    # Patch functions
    import unittest.mock as mock
    with mock.patch('cookiecutter.find.find_template', mock_find_template), \
         mock.patch('cookiecutter.hooks.run_hook_from_repo_dir', mock_run_hook_from_repo_dir), \
         mock.patch('cookiecutter.utils.work_in', mock_work_in), \
         mock.patch('cookiecutter.generate.is_copy_only_path', mock_is_copy_only_path), \
         mock.patch('cookiecutter.generate.generate_file', mock_generate_file):

        # Call function to test
        result = generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)

        # Assertions
        assert isinstance(result, str)
        assert result == "test_output/test_project"


# LLM-generated content at query #24
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    # Setup
    repo_dir = Path("tests/fake-repo-pre")
    context = generate_context(context_file=str(repo_dir / "cookiecutter.json"))
    output_dir = Path("tests/output")
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    # Test
    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    # Verify
    assert Path(project_dir).exists()
    assert (Path(project_dir) / "README.rst").exists()
    assert (Path(project_dir) / "hooks").exists()
    assert (Path(project_dir) / "hooks" / "post_gen_project.py").exists()

    # Cleanup
    shutil.rmtree(project_dir)


# LLM-generated content at query #25
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup a temporary directory and files
    import tempfile
    import os
    from jinja2 import Environment, FileSystemLoader

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test template file
        template_file = os.path.join(tmp_dir, "test_template.txt")
        with open(template_file, "w", encoding="utf-8") as f:
            f.write("Hello {{ name }}!")

        # Create a test context
        context = {"name": "World"}

        # Create a test environment
        env = Environment(loader=FileSystemLoader(tmp_dir))

        # Create a project directory
        project_dir = os.path.join(tmp_dir, "project")
        os.makedirs(project_dir)

        # Call generate_file
        generate_file(project_dir, "test_template.txt", context, env)

        # Verify the output file was created with the correct content
        output_file = os.path.join(project_dir, "test_template.txt")
        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert content == "Hello World!"


# LLM-generated content at query #26
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Setup: Create a temporary cookiecutter.json file with test data
    import tempfile
    import json
    import os

    test_data = {
        "project_name": "Test Project",
        "version": "1.0",
        "description": "A test project",
        "_copy_without_render": ["*.txt"]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    # Test 1: Basic functionality with no extra context
    try:
        context = generate_context(temp_path)
        assert isinstance(context, OrderedDict)
        assert "cookiecutter" in context
        assert context["cookiecutter"]["project_name"] == "Test Project"
    finally:
        os.unlink(temp_path)

    # Test 2: With default context
    default_context = {"version": "2.0"}
    try:
        context = generate_context(temp_path, default_context=default_context)
        assert context["cookiecutter"]["version"] == "2.0"
    finally:
        os.unlink(temp_path)

    # Test 3: With extra context
    extra_context = {"description": "Overridden description"}
    try:
        context = generate_context(temp_path, extra_context=extra_context)
        assert context["cookiecutter"]["description"] == "Overridden description"
    finally:
        os.unlink(temp_path)

    # Test 4: Invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json")
        invalid_path = f.name
    
    try:
        try:
            generate_context(invalid_path)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(invalid_path)

    print("All tests passed!")


# LLM-generated content at query #27
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Setup
    project_dir = "test_project"
    infile = "template.txt"
    context = {"variable": "value"}
    env = Environment(loader=FileSystemLoader("."))
    skip_if_file_exists = False

    # Create a simple template file
    with open(infile, "w") as f:
        f.write("{{ variable }}")

    # Execute
    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # Verify
    expected_file_path = os.path.join(project_dir, infile)
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, "r") as f:
        content = f.read()
        assert content == "value"

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)



# LLM-generated content at query #28
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    # Create a temporary directory for the test
    import tempfile
    import shutil
    import json

    # Set up the test environment
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a template directory
        template_dir = os.path.join(tmp_dir, 'template')
        os.makedirs(template_dir)

        # Create a cookiecutter.json file
        context = {
            "project_name": "Test Project",
            "project_slug": "test_project",
        }
        with open(os.path.join(template_dir, 'cookiecutter.json'), 'w') as f:
            json.dump(context, f)

        # Create a simple template file
        with open(os.path.join(template_dir, 'README.md'), 'w') as f:
            f.write("Project Name: {{ cookiecutter.project_name }}")

        # Generate the project
        output_dir = os.path.join(tmp_dir, 'output')
        project_dir = generate_files(template_dir, context, output_dir)

        # Verify the generated project
        assert os.path.exists(project_dir)
        assert os.path.exists(os.path.join(project_dir, 'README.md'))

        # Verify the content of the README.md file
        with open(os.path.join(project_dir, 'README.md'), 'r') as f:
            content = f.read()
            assert content == "Project Name: Test Project"

    # Clean up
    shutil.rmtree(tmp_dir)


# LLM-generated content at query #29
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid JSON file
    context_file = 'tests/test-context.json'
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert 'project_name' in context['cookiecutter']

    # Test with a non-existent JSON file
    try:
        generate_context('non-existent.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with default and extra context
    default_context = {'cookiecutter': {'project_name': 'Default Project'}}
    extra_context = {'cookiecutter': {'project_name': 'Extra Project'}}
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with invalid JSON file
    invalid_json_file = 'tests/invalid-context.json'
    try:
        generate_context(invalid_json_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    print("All tests passed!")

if __name__ == '__main__':
    test_generate_context()


# LLM-generated content at query #30
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    """Test the generate_context function."""
    # Create a temporary cookiecutter.json file
    with open('cookiecutter.json', 'w', encoding='utf-8') as f:
        json.dump({'project_name': 'Test Project', 'version': '0.1.0'}, f)

    # Test with default context
    context = generate_context()
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'Test Project'
    assert context['cookiecutter']['version'] == '0.1.0'

    # Test with extra context
    extra_context = {'project_name': 'Overridden Project'}
    context = generate_context(extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Overridden Project'

    # Clean up
    os.remove('cookiecutter.json')


# LLM-generated content at query #31
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Create a temporary JSON file for testing
    import tempfile
    import json
    import os

    # Test case 1: Valid JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        json_data = {'key': 'value'}
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    context = generate_context(temp_file_path)
    assert context['cookiecutter'] == json_data, "Test case 1 failed"

    # Clean up
    os.unlink(temp_file_path)

    # Test case 2: Invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        temp_file.write('invalid json')
        temp_file_path = temp_file.name

    try:
        generate_context(temp_file_path)
    except ContextDecodingException as e:
        assert isinstance(e, ContextDecodingException), "Test case 2 failed"

    # Clean up
    os.unlink(temp_file_path)

    # Test case 3: Default context provided
    default_context = {'default_key': 'default_value'}
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        json_data = {'key': 'value'}
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    context = generate_context(temp_file_path, default_context=default_context)
    assert context['cookiecutter']['default_key'] == 'default_value', "Test case 3 failed"

    # Clean up
    os.unlink(temp_file_path)

    # Test case 4: Extra context provided
    extra_context = {'extra_key': 'extra_value'}
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        json_data = {'key': 'value'}
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    context = generate_context(temp_file_path, extra_context=extra_context)
    assert context['cookiecutter']['extra_key'] == 'extra_value', "Test case 4 failed"

    # Clean up
    os.unlink(temp_file_path)

    print("All test cases passed")


# LLM-generated content at query #32
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with default context file
    context = generate_context()
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context

    # Test with custom context file
    custom_context_file = 'custom_context.json'
    with open(custom_context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(custom_context_file)
    assert context['custom_context']['key'] == 'value'
    os.remove(custom_context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(default_context=default_context)
    assert context['cookiecutter']['key'] == 'default_value'

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(extra_context=extra_context)
    assert context['cookiecutter']['key'] == 'extra_value'

    # Test with invalid JSON file
    invalid_json_file = 'invalid.json'
    with open(invalid_json_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')
    try:
        generate_context(invalid_json_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"
    finally:
        os.remove(invalid_json_file)


# LLM-generated content at query #33
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Create a temporary cookiecutter.json file
    with open('cookiecutter.json', 'w', encoding='utf-8') as f:
        json.dump({'project_name': 'My Project'}, f)

    # Test with default context
    context = generate_context()
    assert context['cookiecutter']['project_name'] == 'My Project'

    # Test with extra context
    context = generate_context(extra_context={'project_name': 'New Project'})
    assert context['cookiecutter']['project_name'] == 'New Project'

    # Clean up
    os.remove('cookiecutter.json')


# LLM-generated content at query #34
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    context_file = 'tests/test-context.json'
    default_context = {'cookiecutter': {'project_name': 'Default Project'}}
    extra_context = {'cookiecutter': {'project_name': 'Extra Project'}}

    context = generate_context(context_file, default_context, extra_context)

    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'Extra Project'



# LLM-generated content at query #35
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    project_dir = "test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"_copy_without_render": [], "_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))

    # Create test template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ name }}!")

    # Test rendering
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)

    # Verify output
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello World!"

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    context = {'variable': 'value'}
    output_dir = Path('output')
    environment = Environment()
    
    # Test creating a new directory
    dirname = 'new_dir'
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_path == Path(output_dir, dirname)
    assert created
    assert dir_path.exists()
    
    # Test overwriting an existing directory
    dirname = 'existing_dir'
    existing_dir = Path(output_dir, dirname)
    existing_dir.mkdir(parents=True, exist_ok=True)
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_path == existing_dir
    assert not created
    assert dir_path.exists()
    
    # Test raising OutputDirExistsException when directory exists and overwrite_if_exists is False
    dirname = 'existing_dir_no_overwrite'
    existing_dir = Path(output_dir, dirname)
    existing_dir.mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        assert True
    
    # Test raising EmptyDirNameException when directory name is empty
    dirname = ''
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        assert True
    
    # Clean up
    shutil.rmtree(output_dir)


# LLM-generated content at query #2
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Setup: Create a temporary directory and files for testing
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir = Path(tmpdir) / "template"
        template_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        # Create a simple text file in the template directory
        infile = template_dir / "test_file.txt"
        infile.write_text("Hello, {{ name }}!")

        # Set up context and environment
        context = {"name": "World"}
        env = Environment(loader=FileSystemLoader(template_dir))

        # Execute: Call generate_file
        generate_file(str(project_dir), str(infile), context, env)

        # Verify: Check that the file was generated correctly
        outfile = project_dir / "test_file.txt"
        assert outfile.exists()
        assert outfile.read_text() == "Hello, World!"

        # Cleanup: Temporary directory is automatically cleaned up



# LLM-generated content at query #3
#--------------------------

# Unit test for function apply_overwrites_to_context
def test_apply_overwrites_to_context():
    context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'use_docker': True,
            'features': ['feature1', 'feature2'],
            'database': {
                'engine': 'postgres',
                'port': 5432
            }
        }
    }

    overwrite_context = {
        'project_name': 'New Project',
        'use_docker': 'no',
        'features': ['feature2', 'feature3'],
        'database': {
            'port': 3306
        }
    }

    apply_overwrites_to_context(context['cookiecutter'], overwrite_context)

    assert context['cookiecutter']['project_name'] == 'New Project'
    assert context['cookiecutter']['use_docker'] is False
    assert context['cookiecutter']['features'] == ['feature2', 'feature3']
    assert context['cookiecutter']['database']['port'] == 3306
    assert context['cookiecutter']['database']['engine'] == 'postgres'


# LLM-generated content at query #4
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    repo_dir = Path('tests/fake-repo-pre')
    output_dir = Path('tests/output')
    context = generate_context(context_file='tests/fake-repo-pre/cookiecutter.json')
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(project_dir).exists()
    assert Path(project_dir).is_dir()
    rmtree(project_dir)


# LLM-generated content at query #5
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    # Test case 1: Normal operation with default parameters
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-pre/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        project_dir = generate_files(repo_dir, output_dir=output_dir)
        assert Path(project_dir).exists()
    finally:
        shutil.rmtree(output_dir)

    # Test case 2: Overwrite existing directory
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-pre/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
        project_dir = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert Path(project_dir).exists()
    finally:
        shutil.rmtree(output_dir)

    # Test case 3: Skip existing files
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-pre/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
        project_dir = generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        assert Path(project_dir).exists()
    finally:
        shutil.rmtree(output_dir)

    # Test case 4: Keep project on failure
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-pre/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        project_dir = generate_files(repo_dir, output_dir=output_dir, keep_project_on_failure=True)
        assert Path(project_dir).exists()
    except Exception:
        assert Path(project_dir).exists()
        shutil.rmtree(output_dir)

    # Test case 5: Undefined variable in template
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-undef-var/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
    except UndefinedVariableInTemplate as e:
        assert str(e) == "Unable to create file '{{ cookiecutter.undefined_var }}.txt'"
    finally:
        shutil.rmtree(output_dir)

    # Test case 6: Empty directory name
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-empty-dir/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
    except EmptyDirNameException as e:
        assert str(e) == "Error: directory name is empty"
    finally:
        shutil.rmtree(output_dir)

    # Test case 7: Output directory exists
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-pre/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
        generate_files(repo_dir, output_dir=output_dir)
    except OutputDirExistsException as e:
        assert str(e) == f'Error: "{output_dir}/test-repo-pre" directory already exists'
    finally:
        shutil.rmtree(output_dir)

    # Test case 8: Context decoding error
    repo_dir = Path(__file__).parent.parent / "tests/test-repo-invalid-json/"
    output_dir = Path(__file__).parent.parent / "tests/test-output/"
    try:
        generate_files(repo_dir, output_dir=output_dir)
    except ContextDecodingException as e:
        assert str(e).startswith("JSON decoding error while loading")
    finally:
        shutil.rmtree(output_dir)


# LLM-generated content at query #6
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    """Test the generate_context function."""
    # Test with a valid JSON file
    context_file = 'tests/test-context.json'
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert 'project_name' in context['cookiecutter']

    # Test with a non-existent JSON file
    try:
        generate_context('non-existent.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with an invalid JSON file
    try:
        generate_context('tests/invalid-context.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'


# LLM-generated content at query #7
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    # Create a temporary directory for testing
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory for the template
    template_dir = tempfile.mkdtemp()
    try:
        # Create a simple template structure
        (Path(template_dir) / 'cookiecutter.json').write_text(
            '{"project_name": "Test Project", "repo_name": "{{ cookiecutter.project_name.lower().replace(\' \', \'-\') }}"}'
        )
        (Path(template_dir) / 'README.md').write_text('# {{ cookiecutter.project_name }}')
        (Path(template_dir) / 'static').mkdir()
        (Path(template_dir) / 'static' / 'logo.png').write_text('PNG image data')

        # Create a temporary output directory
        output_dir = tempfile.mkdtemp()
        try:
            # Test basic generation
            context = {'cookiecutter': {'project_name': 'Test Project'}}
            project_path = generate_files(
                template_dir,
                context=context,
                output_dir=output_dir,
                overwrite_if_exists=True
            )
            assert Path(project_path).exists()
            assert (Path(project_path) / 'README.md').exists()
            assert (Path(project_path) / 'static' / 'logo.png').exists()

            # Verify rendered content
            readme_content = (Path(project_path) / 'README.md').read_text()
            assert readme_content == '# Test Project'

            # Verify directory name was rendered
            assert Path(project_path).name == 'test-project'

            # Test skip_if_file_exists
            (Path(project_path) / 'existing_file').write_text('original')
            generate_files(
                template_dir,
                context=context,
                output_dir=output_dir,
                skip_if_file_exists=True
            )
            assert (Path(project_path) / 'existing_file').read_text() == 'original'

            # Test overwrite_if_exists=False with existing directory
            try:
                generate_files(
                    template_dir,
                    context=context,
                    output_dir=output_dir,
                    overwrite_if_exists=False
                )
                assert False, "Should have raised OutputDirExistsException"
            except OutputDirExistsException:
                pass

            # Test with hooks (mock test since we can't easily test real hooks)
            hook_dir = Path(template_dir) / 'hooks'
            hook_dir.mkdir()
            (hook_dir / 'pre_gen_project.py').write_text('print("Pre-gen hook")')
            (hook_dir / 'post_gen_project.py').write_text('print("Post-gen hook")')
            generate_files(
                template_dir,
                context=context,
                output_dir=output_dir,
                overwrite_if_exists=True,
                accept_hooks=True
            )

        finally:
            shutil.rmtree(output_dir)
    finally:
        shutil.rmtree(template_dir)


# LLM-generated content at query #8
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test case 1: Normal context generation
    context_file = 'test_cookiecutter.json'
    default_context = {'key1': 'value1'}
    extra_context = {'key2': 'value2'}
    context = generate_context(context_file, default_context, extra_context)
    assert context['test_cookiecutter']['key1'] == 'value1'
    assert context['test_cookiecutter']['key2'] == 'value2'

    # Test case 2: JSON decoding error
    invalid_context_file = 'invalid_cookiecutter.json'
    try:
        generate_context(invalid_context_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test case 3: Invalid default context
    context_file = 'test_cookiecutter.json'
    invalid_default_context = {'key1': 'invalid_value'}
    try:
        generate_context(context_file, invalid_default_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

    # Test case 4: Invalid extra context
    invalid_extra_context = {'key2': 'invalid_value'}
    try:
        generate_context(context_file, None, invalid_extra_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #9
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    test_project_dir = "test_project"
    test_infile = "test_template.txt"
    test_context = {"variable": "value"}
    test_env = Environment(loader=FileSystemLoader("."))
    
    # Create a test template file
    with open(test_infile, "w", encoding="utf-8") as f:
        f.write("This is a test template with {{ variable }}")
    
    # Test
    generate_file(test_project_dir, test_infile, test_context, test_env)
    
    # Verify
    expected_outfile = os.path.join(test_project_dir, test_infile)
    assert os.path.exists(expected_outfile)
    with open(expected_outfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "This is a test template with value"
    
    # Cleanup
    os.remove(test_infile)
    shutil.rmtree(test_project_dir)


# LLM-generated content at query #10
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    project_dir = "test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))
    
    # Create a test template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ name }}!")
    
    # Test rendering
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    
    # Verify the output file
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello World!"
    
    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #11
#--------------------------

# Unit test for function is_copy_only_path
def test_is_copy_only_path():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'images/*']
        }
    }
    assert is_copy_only_path('example.txt', context) == True
    assert is_copy_only_path('images/photo.jpg', context) == True
    assert is_copy_only_path('scripts/script.py', context) == False
    assert is_copy_only_path('README.md', context) == False



# LLM-generated content at query #12
#--------------------------

# Unit test for function is_copy_only_path
def test_is_copy_only_path():
    """Test the is_copy_only_path function."""
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/file.txt', context) is True
    assert is_copy_only_path('docs/subdir/file.txt', context) is True
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('templates/file.txt', context) is False
    assert is_copy_only_path('file.txt', {}) is False


# LLM-generated content at query #13
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test generate_file function."""
    # Setup test environment
    project_dir = Path("test_project")
    infile = "test_template.txt"
    context = {"cookiecutter": {"project_name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))

    # Create a template file
    with open(infile, "w") as f:
        f.write("Project Name: {{ cookiecutter.project_name }}")

    # Execute the function
    generate_file(project_dir, infile, context, env)

    # Verify the output file
    outfile = Path(project_dir) / "test_template.txt"
    assert outfile.exists()
    with open(outfile) as f:
        content = f.read()
    assert content == "Project Name: Test Project"

    # Clean up
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #14
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    # Create a temporary directory for testing
    import tempfile
    temp_dir = tempfile.mkdtemp()

    # Define a simple context and environment
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    environment = Environment(loader=FileSystemLoader(temp_dir))

    # Test creating a new directory
    dirname = '{{ cookiecutter.project_name }}'
    output_dir = Path(temp_dir)
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert created
    assert dir_path.exists()
    assert dir_path.name == 'TestProject'

    # Test overwriting an existing directory
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert not created
    assert dir_path.exists()
    assert dir_path.name == 'TestProject'

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #15
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    """Test the render_and_create_dir function."""
    # Setup
    dirname = "test_dir"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = Path("test_output")
    environment = Environment()
    overwrite_if_exists = False

    # Test creating a new directory
    dir_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert dir_path == output_dir / dirname
    assert created is True
    assert dir_path.exists()
    dir_path.rmdir()

    # Test overwriting an existing directory
    dir_path.mkdir()
    dir_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert dir_path == output_dir / dirname
    assert created is False
    assert dir_path.exists()
    dir_path.rmdir()

    # Test raising OutputDirExistsException when overwrite is False
    dir_path.mkdir()
    try:
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )
    except OutputDirExistsException:
        pass
    else:
        assert False, "Expected OutputDirExistsException"
    finally:
        dir_path.rmdir()

    # Test raising EmptyDirNameException when dirname is empty
    try:
        render_and_create_dir(
            "", context, output_dir, environment, overwrite_if_exists
        )
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException"


# LLM-generated content at query #16
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test case 1: Basic functionality
    context_file = 'tests/unit/test_context.json'
    default_context = {'project_name': 'Default Project'}
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(context_file, default_context, extra_context)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test case 2: File not found
    try:
        generate_context('nonexistent.json')
    except FileNotFoundError as e:
        assert str(e) == "[Errno 2] No such file or directory: 'nonexistent.json'"

    # Test case 3: Invalid JSON
    try:
        generate_context('tests/unit/invalid.json')
    except ContextDecodingException as e:
        assert "JSON decoding error while loading" in str(e)

    # Test case 4: No default or extra context
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'Test Project'

    # Test case 5: Invalid default context
    try:
        generate_context(context_file, {'invalid_key': 'Invalid Value'})
    except ValueError as e:
        assert "Invalid default received" in str(e)

    # Test case 6: Invalid extra context
    try:
        generate_context(context_file, None, {'invalid_key': 'Invalid Value'})
    except ValueError as e:
        assert "Invalid default received" in str(e)

    # Test case 7: Default context overwrite
    context = generate_context(context_file, {'project_name': 'Default Project'})
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test case 8: Extra context overwrite
    context = generate_context(context_file, None, {'project_name': 'Extra Project'})
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test case 9: Default and extra context overwrite
    context = generate_context(context_file, {'project_name': 'Default Project'}, {'project_name': 'Extra Project'})
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test case 10: Empty context file
    try:
        generate_context('tests/unit/empty.json')
    except ContextDecodingException as e:
        assert "JSON decoding error while loading" in str(e)


# LLM-generated content at query #17
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    """Test the render_and_create_dir function."""
    # Setup
    dirname = "test_dir"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = Path("test_output")
    environment = Environment()
    overwrite_if_exists = False

    # Test creating a new directory
    dir_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert dir_path == Path(output_dir, dirname)
    assert created is True
    assert dir_path.exists()
    dir_path.rmdir()

    # Test directory already exists and overwrite is False
    dir_path.mkdir()
    try:
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists
        )
    except OutputDirExistsException:
        pass
    else:
        assert False, "Expected OutputDirExistsException"
    dir_path.rmdir()

    # Test directory already exists and overwrite is True
    dir_path.mkdir()
    dir_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, True
    )
    assert dir_path == Path(output_dir, dirname)
    assert created is False
    assert dir_path.exists()
    dir_path.rmdir()

    # Test empty directory name
    try:
        render_and_create_dir("", context, output_dir, environment, overwrite_if_exists)
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException"

    # Cleanup
    if output_dir.exists():
        output_dir.rmdir()


# LLM-generated content at query #18
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with default context and extra context
    default_context = {'key1': 'value1'}
    extra_context = {'key2': 'value2'}
    context = generate_context(default_context=default_context, extra_context=extra_context)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['key1'] == 'value1'
    assert context['cookiecutter']['key2'] == 'value2'

    # Test with invalid JSON file
    try:
        generate_context(context_file='invalid.json')
        assert False, "Expected ContextDecodingException"
    except ContextDecodingException:
        pass

    # Test with valid JSON file
    with open('valid.json', 'w') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(context_file='valid.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['key'] == 'value'
    os.remove('valid.json')


# LLM-generated content at query #19
#--------------------------

# Unit test for function apply_overwrites_to_context
def test_apply_overwrites_to_context():
    context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'version': '1.0',
            'features': ['feature1', 'feature2'],
            'settings': {
                'debug': True,
                'log_level': 'info'
            }
        }
    }

    overwrite_context = {
        'project_name': 'Overwritten Project',
        'features': ['feature3'],
        'settings': {
            'log_level': 'debug'
        }
    }

    apply_overwrites_to_context(context, overwrite_context)

    assert context['cookiecutter']['project_name'] == 'Overwritten Project'
    assert context['cookiecutter']['features'] == ['feature3']
    assert context['cookiecutter']['settings']['log_level'] == 'debug'
    assert context['cookiecutter']['settings']['debug'] is True


# LLM-generated content at query #20
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid JSON file
    context_file = 'tests/fixtures/fake-repo/cookiecutter.json'
    context = generate_context(context_file)
    assert isinstance(context, dict)
    assert 'cookiecutter' in context
    assert 'full_name' in context['cookiecutter']
    assert 'email' in context['cookiecutter']

    # Test with a default context
    default_context = {'cookiecutter': {'full_name': 'Default Name'}}
    context = generate_context(context_file, default_context=default_context)
    assert context['cookiecutter']['full_name'] == 'Default Name'

    # Test with an extra context
    extra_context = {'cookiecutter': {'full_name': 'Extra Name'}}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['full_name'] == 'Extra Name'

    # Test with a non-existent JSON file
    try:
        generate_context('nonexistent.json')
    except ContextDecodingException as e:
        assert isinstance(e, ContextDecodingException)

    # Test with an invalid JSON file
    try:
        generate_context('tests/fixtures/invalid-repo/cookiecutter.json')
    except ContextDecodingException as e:
        assert isinstance(e, ContextDecodingException)


# LLM-generated content at query #21
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test case 1: Simple context file
    context_file = 'tests/fake-repo/cookiecutter.json'
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context

    # Test case 2: Context file with default context
    default_context = {"project_name": "Test Project"}
    context = generate_context(context_file, default_context=default_context)
    assert context['cookiecutter']['project_name'] == "Test Project"

    # Test case 3: Context file with extra context
    extra_context = {"project_name": "Extra Project"}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == "Extra Project"

    # Test case 4: Invalid context file path
    invalid_context_file = 'tests/fake-repo/invalid.json'
    try:
        generate_context(invalid_context_file)
    except ContextDecodingException:
        assert True
    else:
        assert False

    # Test case 5: Invalid JSON in context file
    invalid_json_file = 'tests/fake-repo/invalid-json.json'
    try:
        generate_context(invalid_json_file)
    except ContextDecodingException:
        assert True
    else:
        assert False


# LLM-generated content at query #22
#--------------------------

# Unit test for function render_and_create_dir
def test_render_and_create_dir():
    # Initialize test environment
    context = {'cookiecutter': {'name': 'test_project'}}
    output_dir = Path('/tmp')
    environment = Environment(loader=FileSystemLoader(['.']))

    # Test case 1: Normal case
    dirname = "{{ cookiecutter.name }}"
    rendered_dir, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert rendered_dir == Path('/tmp/test_project')
    assert created == True
    assert rendered_dir.exists()

    # Test case 2: Empty directory name
    dirname = ""
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

    # Test case 3: Directory already exists
    dirname = "{{ cookiecutter.name }}"
    rendered_dir, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert rendered_dir == Path('/tmp/test_project')
    assert created == False
    assert rendered_dir.exists()

    # Test case 4: Directory already exists with overwrite_if_exists
    dirname = "{{ cookiecutter.name }}"
    rendered_dir, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert rendered_dir == Path('/tmp/test_project')
    assert created == False
    assert rendered_dir.exists()

    # Clean up
    shutil.rmtree(rendered_dir)


# LLM-generated content at query #23
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    # Test case 1: Basic functionality with no hooks and no existing output directory
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "TestProject"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)

    # Clean up after test case 1
    shutil.rmtree(output_dir)

    # Test case 2: Overwrite existing directory
    os.makedirs(output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)

    # Clean up after test case 2
    shutil.rmtree(output_dir)

    # Test case 3: Skip existing files
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)

    # Clean up after test case 3
    shutil.rmtree(output_dir)

    # Test case 4: With hooks and keep project on failure
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True, keep_project_on_failure=True)
    assert os.path.exists(result)

    # Clean up after test case 4
    shutil.rmtree(output_dir)

    # Test case 5: Undefined variable in template
    context = {"cookiecutter": {"project_name": "{{ undefined_variable }}"}}
    try:
        generate_files(repo_dir, context, output_dir)
    except UndefinedVariableInTemplate:
        assert True
    else:
        assert False

    # Clean up after test case 5
    shutil.rmtree(output_dir)


# LLM-generated content at query #24
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Create a temporary directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a template file
        template_file = Path(tmpdir) / "template.txt"
        template_file.write_text("Hello, {{ name }}!")

        # Create a context dictionary
        context = {"name": "World"}

        # Create a Jinja2 environment
        env = Environment(loader=FileSystemLoader(tmpdir))

        # Generate the file
        generate_file(tmpdir, "template.txt", context, env)

        # Check if the file was generated correctly
        generated_file = Path(tmpdir) / "template.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Hello, World!"



# LLM-generated content at query #25
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Setup
    project_dir = "test_project"
    infile = "template.txt"
    context = {"variable": "value"}
    env = Environment(loader=FileSystemLoader("."))

    # Create a simple template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("This is a template with {{ variable }}.")

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    with open(os.path.join(project_dir, infile), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "This is a template with value."

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)



# LLM-generated content at query #26
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Create a temporary directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary file in the directory
        with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
            f.write('Hello {{ name }}')

        # Create a context dictionary
        context = {'name': 'World'}

        # Create a Jinja2 environment
        env = Environment(loader=FileSystemLoader(tmpdir))

        # Generate the file
        generate_file(tmpdir, 'test.txt', context, env)

        # Check if the file was generated correctly
        with open(os.path.join(tmpdir, 'test.txt'), 'r') as f:
            assert f.read() == 'Hello World'



# LLM-generated content at query #27
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    """Test the generate_files function."""
    repo_dir = Path(__file__).parent / 'tests' / 'fake-repo-tmpl'
    context = {'cookiecutter': {'project_name': 'fake-project'}}
    output_dir = Path(__file__).parent / 'tests' / 'fake-output-dir'
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    # Test generating files
    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    # Check if the project directory is created
    assert Path(project_dir).exists()

    # Clean up
    shutil.rmtree(output_dir)


# LLM-generated content at query #28
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(context_file)
    assert context['test_cookiecutter']['key'] == 'value'
    os.remove(context_file)

    # Test with an invalid JSON file
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')
    try:
        generate_context(context_file)
        assert False, "Should have raised ContextDecodingException"
    except ContextDecodingException:
        pass
    finally:
        os.remove(context_file)

    # Test with default_context
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(context_file, default_context={'key': 'new_value'})
    assert context['test_cookiecutter']['key'] == 'new_value'
    os.remove(context_file)

    # Test with extra_context
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(context_file, extra_context={'key': 'extra_value'})
    assert context['test_cookiecutter']['key'] == 'extra_value'
    os.remove(context_file)


# LLM-generated content at query #29
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    # Setup
    project_dir = "test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))
    
    # Create a dummy template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("This is a test template for {{ cookiecutter.name }}")
    
    # Test generating file
    generate_file(project_dir, infile, context, env)
    
    # Assertions
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "This is a test template for Test Project"
    
    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #30
#--------------------------

# Unit test for function generate_files
def test_generate_files():
    # Test case 1: Basic functionality test
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
        }
    }
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

    # Clean up
    shutil.rmtree(output_dir)

    # Test case 2: Test with existing project directory and overwrite_if_exists set to False
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
        }
    }
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    # Create the output directory first
    os.makedirs(output_dir / "test_project")

    try:
        generate_files(
            repo_dir,
            context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure,
        )
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

    # Clean up
    shutil.rmtree(output_dir)

    # Test case 3: Test with skip_if_file_exists set to True
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
        }
    }
    overwrite_if_exists = True
    skip_if_file_exists = True
    accept_hooks = True
    keep_project_on_failure = False

    # Create the output directory first
    os.makedirs(output_dir / "test_project")

    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

    # Clean up
    shutil.rmtree(output_dir)

    # Test case 4: Test with hooks disabled
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
        }
    }
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = False

    project_dir = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

    # Clean up
    shutil.rmtree(output_dir)

    # Test case 5: Test with keep_project_on_failure set to True
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
        }
    }
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = True

    # Create a faulty context to trigger an error
    faulty_context = {
        "cookiecutter": {
            "full_name": "Test User",
            "email": "test@example.com",
            "project_name": "Test Project",
            "repo_name": "test_project",
            "_copy_without_render": [],
            "undefined_variable": "{{ undefined_variable }}",
        }
    }

    try:
        generate_files(
            repo_dir,
            faulty_context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure,
        )
        assert False, "Expected UndefinedVariableInTemplate"
    except UndefinedVariableInTemplate:
        # Ensure the project directory is not deleted
        assert os.path.exists(output_dir / "test_project")

    # Clean up
    shutil.rmtree(output_dir)


# LLM-generated content at query #31
#--------------------------

# Unit test for function generate_context
def test_generate_context():
    # Test with a valid JSON file
    context_file = 'tests/test-context.json'
    context = generate_context(context_file)
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context
    assert 'project_name' in context['cookiecutter']

    # Test with a non-existent JSON file
    try:
        generate_context('non-existent.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with an invalid JSON file
    try:
        generate_context('tests/invalid-context.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(context_file, None, extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    print("All tests passed!")

test_generate_context()


# LLM-generated content at query #32
#--------------------------

# Unit test for function generate_file
def test_generate_file():
    """Test the generate_file function."""
    # Setup
    project_dir = "test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"_copy_without_render": [], "_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))
    
    # Create test template file
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ name }}!")
    
    # Test rendering
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    
    # Verify output
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello World!"
    
    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


