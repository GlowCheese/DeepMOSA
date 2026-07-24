####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test overwrite with new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list overwrite (choice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test list overwrite with invalid choice (should raise ValueError)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test multichoice list overwrite
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test multichoice list overwrite with invalid choice (should raise ValueError)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test dict overwrite
    context = {'var1': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'var1': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'value2'

    # Test dict overwrite with new key (should be added)
    context = {'var1': {'key1': 'value1'}}
    overwrite_context = {'var1': {'key1': 'new_value1', 'key2': 'value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'value2'

    # Test boolean overwrite with string
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean overwrite with invalid string (should raise ValueError)
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_bool" in str(e)

    # Test nested dict overwrite
    context = {'var1': {'nested': {'key1': 'value1'}}}
    overwrite_context = {'var1': {'nested': {'key1': 'new_value1'}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested']['key1'] == 'new_value1'


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    dirname = '{{ cookiecutter.project_name }}'

    # Test successful directory creation
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=False
    )
    assert result_dir == output_dir / 'test_project'
    assert created is True
    assert result_dir.exists()

    # Test overwrite_if_exists=True
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert result_dir == output_dir / 'test_project'
    assert created is False  # Directory already exists

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, environment)

    # Test undefined variable in template
    bad_context = {'wrong_key': 'value'}
    with pytest.raises(UndefinedVariableInTemplate):
        render_and_create_dir(
            dirname, bad_context, output_dir, environment, overwrite_if_exists=False
        )

    # Test existing directory without overwrite
    existing_dir = output_dir / 'existing'
    existing_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            'existing', context, output_dir, environment, overwrite_if_exists=False
        )

    # Cleanup
    rmtree(output_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"test_var": "test_value"}
    output_dir = Path("/tmp/test_output")
    environment = Environment()
    overwrite_if_exists = False

    # Test normal case
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path == output_dir / "test_dir"
    assert created is True
    assert (output_dir / "test_dir").exists()

    # Test with template in dirname
    dirname = "{{ test_var }}"
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path == output_dir / "test_value"
    assert created is True
    assert (output_dir / "test_value").exists()

    # Test overwrite existing directory
    overwrite_if_exists = True
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path == output_dir / "test_value"
    assert created is False  # Directory already existed

    # Test empty dirname
    dirname = ""
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists
        )

    # Test undefined variable in template
    dirname = "{{ undefined_var }}"
    with pytest.raises(UndefinedVariableInTemplate):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists
        )

    # Cleanup
    if (output_dir / "test_dir").exists():
        rmtree(output_dir / "test_dir")
    if (output_dir / "test_value").exists():
        rmtree(output_dir / "test_value")


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list choice variable
    context = {'choice_var': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'choice_var': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choice_var'] == ['choice2', 'choice1', 'choice3']

    # Test invalid choice variable
    context = {'choice_var': ['choice1', 'choice2']}
    overwrite_context = {'choice_var': 'invalid_choice'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test multichoice variable
    context = {'multi_choice': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'multi_choice': ['choice2', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['multi_choice'] == ['choice2', 'choice3']

    # Test invalid multichoice variable
    context = {'multi_choice': ['choice1', 'choice2']}
    overwrite_context = {'multi_choice': ['choice1', 'invalid_choice']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test boolean variable
    context = {'bool_var': True}
    overwrite_context = {'bool_var': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['bool_var'] is True

    # Test invalid boolean variable
    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'invalid_bool'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_bool" in str(e)

    # Test dictionary variable
    context = {'dict_var': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'dict_var': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['dict_var'] == {'key1': 'new_value1', 'key2': 'value2'}

    # Test new dictionary variable key
    context = {'dict_var': {'key1': 'value1'}}
    overwrite_context = {'dict_var': {'key1': 'new_value1', 'key2': 'value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['dict_var'] == {'key1': 'new_value1', 'key2': 'value2'}


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = False

    # Test normal case
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_dir.exists()
    assert created
    assert result_dir.name == "test_dir"

    # Test with template rendering
    dirname = "{{ test }}_dir"
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_dir.exists()
    assert created
    assert result_dir.name == "value_dir"

    # Test overwrite case
    overwrite_if_exists = True
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_dir.exists()
    assert not created  # Should not create new since it exists

    # Test empty dirname case
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment, overwrite_if_exists)

    # Cleanup
    shutil.rmtree(result_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_files(mocker, tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    output_dir = tmp_path / "output"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
            '_copy_without_render': ['*.bin']
        }
    }

    # Create template structure
    template_dir = repo_dir / "{{cookiecutter.project_slug}}"
    template_dir.mkdir(parents=True)
    (template_dir / "test.txt").write_text("Hello {{cookiecutter.project_name}}!")
    (template_dir / "binary.bin").write_bytes(b'\x00\x01\x02\x03')

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(template_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(template_dir))))
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Run the function
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )

    # Assertions
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test.txt").exists()
    assert (output_dir / "test_project" / "test.txt").read_text() == "Hello test_project!"
    assert (output_dir / "test_project" / "binary.bin").exists()
    assert (output_dir / "test_project" / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=str(output_dir),
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=True,
        keep_project_on_failure=False
    )

    # Test error handling for undefined variable
    bad_context = {'cookiecutter': {'project_name': 'test'}}
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context=bad_context,
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            skip_if_file_exists=False,
            accept_hooks=True,
            keep_project_on_failure=False
        )

    # Test error handling for existing output directory
    with pytest.raises(OutputDirExistsException):
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False,
            skip_if_file_exists=False,
            accept_hooks=True,
            keep_project_on_failure=False
        )


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = "/tmp/test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, "w") as f:
        f.write("Hello, {{ cookiecutter.name }}!")

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r") as f:
        content = f.read()
    assert content == "Hello, test!"

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list overwrite (choice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice overwrite
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test invalid multichoice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test nested dict overwrite
    context = {'var1': {'nested1': 'value1', 'nested2': 'value2'}}
    overwrite_context = {'var1': {'nested1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'value2'

    # Test boolean overwrite
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test invalid boolean overwrite
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_boolean'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test nested dict with new variable
    context = {'var1': {'nested1': 'value1'}}
    overwrite_context = {'var1': {'nested1': 'new_value1', 'nested2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'new_value2'


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'key': 'default_value'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}
    os.remove(context_file)

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)


# LLM-generated content at query #11
#--------------------------

```python
def test_is_copy_only_path():
    # Test when path matches a pattern in context
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/readme.md', context) is True

    # Test when path does not match any pattern
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

    # Test when context does not have _copy_without_render key
    context_no_key = {'cookiecutter': {}}
    assert is_copy_only_path('file.txt', context_no_key) is False

    # Test with empty context
    empty_context = {}
    assert is_copy_only_path('file.txt', empty_context) is False


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context)
    assert context == {'test_context': {'key': 'default_value'}}

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, None, extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    # Clean up
    os.remove(context_file)


# LLM-generated content at query #13
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test new variable on first level (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test new variable in deeper level (should be added)
    context = {'var1': {'nested_var1': 'value1'}}
    overwrite_context = {'var1': {'nested_var1': 'new_value1', 'nested_var2': 'value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested_var1'] == 'new_value1'
    assert context['var1']['nested_var2'] == 'value2'

    # Test multichoice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice2', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice3']

    # Test invalid multichoice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice4']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test choice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test invalid choice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice4'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test boolean variable
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    context = {'var1': False}
    overwrite_context = {'var1': 'no'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is False

    # Test invalid boolean variable
    context = {'var1': True}
    overwrite_context = {'var1': 'invalid'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #14
#--------------------------

```python
def test_is_copy_only_path():
    # Test when path matches a pattern in context
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/index.md', context) is True

    # Test when path does not match any pattern
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

    # Test when context does not have '_copy_without_render'
    context_no_pattern = {
        'cookiecutter': {
            'some_other_key': 'value'
        }
    }
    assert is_copy_only_path('file.txt', context_no_pattern) is False

    # Test when context is empty
    empty_context = {}
    assert is_copy_only_path('file.txt', empty_context) is False

    # Test with nested directory structure
    context_nested = {
        'cookiecutter': {
            '_copy_without_render': ['static/*']
        }
    }
    assert is_copy_only_path('static/images/logo.png', context_nested) is True
    assert is_copy_only_path('static/js/script.js', context_nested) is True
    assert is_copy_only_path('dynamic/content.html', context_nested) is False


# LLM-generated content at query #15
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("test content")

    context = {"project_name": "test_project"}
    output_dir = tmp_path / "output"

    # Test basic generation
    result = generate_files(repo_dir, context, output_dir)
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "file.txt").exists()
    assert (output_dir / "test_project" / "file.txt").read_text() == "test content"

    # Test overwrite_if_exists
    generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert (output_dir / "test_project" / "file.txt").exists()

    # Test skip_if_file_exists
    (output_dir / "test_project" / "file.txt").write_text("modified content")
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert (output_dir / "test_project" / "file.txt").read_text() == "modified content"

    # Test with hooks (create dummy hook files)
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
    (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")
    generate_files(repo_dir, context, tmp_path / "output_with_hooks", accept_hooks=True)
    assert (tmp_path / "output_with_hooks" / "test_project" / "file.txt").exists()

    # Test keep_project_on_failure
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("raise Exception('hook failed')")
    with pytest.raises(Exception):
        generate_files(repo_dir, context, tmp_path / "output_failure", keep_project_on_failure=True)
    assert (tmp_path / "output_failure" / "test_project").exists()


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mocks/cookiecutter.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'My Project'

    # Test with default context
    default_context = {'project_name': 'Default Project'}
    context = generate_context('tests/mocks/cookiecutter.json', default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context('tests/mocks/cookiecutter.json', extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mocks/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mocks/nonexistent.json')


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "tests/mocks/valid-template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "tests/mocks/output"

    # Execute
    result = generate_files(repo_dir, context, output_dir)

    # Verify
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert os.path.basename(result) == "test_project"
    assert os.path.exists(os.path.join(result, "README.md"))
    assert os.path.exists(os.path.join(result, "setup.py"))


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'key': 'default_value'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}
    os.remove(context_file)

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    test_dir = Path(tempfile.mkdtemp())
    template_dir = test_dir / "template"
    template_dir.mkdir()
    output_dir = test_dir / "output"
    output_dir.mkdir()

    # Create a test template file
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello, {{ name }}!")

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "name": "World"
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test file generation
    generate_file(
        project_dir=str(output_dir),
        infile="test.txt",
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    # Verify output
    output_file = output_dir / "test.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

    # Test binary file handling
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    context["cookiecutter"]["_copy_without_render"] = ["*.bin"]

    generate_file(
        project_dir=str(output_dir),
        infile="binary.bin",
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    output_binary = output_dir / "binary.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == b"\x00\x01\x02\x03"

    # Test skip_if_file_exists
    generate_file(
        project_dir=str(output_dir),
        infile="test.txt",
        context=context,
        env=env,
        skip_if_file_exists=True
    )
    # File should still exist and not be modified
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

    # Cleanup
    shutil.rmtree(test_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Assert
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, {{ name }}!'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_files():
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock template directory
        template_dir = temp_path / "template"
        template_dir.mkdir()

        # Create a mock cookiecutter.json
        cookiecutter_json = template_dir / "cookiecutter.json"
        cookiecutter_json.write_text('{"project_name": "test_project", "project_slug": "{{ cookiecutter.project_name.lower().replace(\' \', \'_') }}"}')

        # Create a mock template file
        template_file = template_dir / "{{ cookiecutter.project_slug }}.txt"
        template_file.write_text("Hello, {{ cookiecutter.project_name }}!")

        # Create a mock copy-only file
        copy_file = template_dir / "copy_only.txt"
        copy_file.write_text("This file should be copied without rendering.")

        # Create a mock context
        context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "_copy_without_render": ["copy_only.txt"]
            }
        }

        # Create a mock output directory
        output_dir = temp_path / "output"

        # Test
        result = generate_files(
            repo_dir=template_dir,
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=True,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )

        # Assertions
        # Check if the project directory was created
        assert (output_dir / "test_project").exists()

        # Check if the rendered file exists and has the correct content
        rendered_file = output_dir / "test_project" / "test_project.txt"
        assert rendered_file.exists()
        assert rendered_file.read_text() == "Hello, Test Project!"

        # Check if the copy-only file exists and has the correct content
        copy_only_file = output_dir / "test_project" / "copy_only.txt"
        assert copy_only_file.exists()
        assert copy_only_file.read_text() == "This file should be copied without rendering."

        # Check if the function returns the correct project directory path
        assert result == str(output_dir / "test_project")


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, 'test_template.txt')
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "path/to/template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "path/to/output"
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = True

    # Mock the necessary functions and objects
    with patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_dir, \
         patch('cookiecutter.generate.work_in') as mock_work_in, \
         patch('os.walk') as mock_walk, \
         patch('cookiecutter.generate.is_copy_only_path') as mock_is_copy, \
         patch('cookiecutter.generate.generate_file') as mock_gen_file, \
         patch('cookiecutter.generate.shutil.copytree') as mock_copytree, \
         patch('cookiecutter.generate.shutil.rmtree') as mock_rmtree:

        # Configure mocks
        mock_find_template.return_value = "path/to/template"
        mock_create_env.return_value = Environment(loader=FileSystemLoader('.'))
        mock_render_dir.return_value = ("path/to/output/test_project", True)
        mock_walk.return_value = [
            ('.', ['dir1', 'dir2'], ['file1.txt', 'file2.txt']),
            ('dir1', [], ['file3.txt'])
        ]
        mock_is_copy.side_effect = lambda x, _: x == 'dir1' or x == 'file1.txt'

        # Execute
        result = generate_files(
            repo_dir,
            context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure
        )

        # Assert
        assert result == "path/to/output/test_project"
        mock_find_template.assert_called_once_with(repo_dir, mock_create_env.return_value)
        mock_render_dir.assert_called_once_with(
            "template",
            context,
            output_dir,
            mock_create_env.return_value,
            overwrite_if_exists
        )
        mock_work_in.assert_called_once_with("path/to/template")
        mock_walk.assert_called_once_with('.')
        mock_is_copy.assert_called()
        mock_gen_file.assert_called()
        mock_copytree.assert_called_once()
        mock_rmtree.assert_not_called()


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'file.txt'), 'w') as f:
            f.write('test content')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test', 'file.txt'))
        assert result == os.path.join(output_dir, 'test')

    # Test with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(os.path.join(output_dir, 'test'))

        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert os.path.exists(os.path.join(output_dir, 'test'))
        assert result == os.path.join(output_dir, 'test')

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'file.txt'), 'w') as f:
            f.write('test content')

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(os.path.join(output_dir, 'test'))
        with open(os.path.join(output_dir, 'test', 'file.txt'), 'w') as f:
            f.write('existing content')

        result = generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        with open(os.path.join(output_dir, 'test', 'file.txt')) as f:
            assert f.read() == 'existing content'
        assert result == os.path.join(output_dir, 'test')

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, 'hooks'))
        with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('print("pre hook")')
        with open(os.path.join(repo_dir, 'hooks', 'post_gen_project.py'), 'w') as f:
            f.write('print("post hook")')
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)
        assert os.path.exists(os.path.join(output_dir, 'test'))
        assert result == os.path.join(output_dir, 'test')

    # Test with context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', '{{cookiecutter.project_name}}.txt'), 'w') as f:
            f.write('test content')

        output_dir = os.path.join(tmpdir, 'output')
        context = {'cookiecutter': {'project_name': 'myproject'}}
        result = generate_files(repo_dir, context=context, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'myproject', 'myproject.txt'))
        assert result == os.path.join(output_dir, 'myproject')

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test', '_copy_without_render': ['*.bin']}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'file.bin'), 'w') as f:
            f.write('binary content')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test', 'file.bin'))
        assert result == os.path.join(output_dir, 'test')

    # Test with exception for existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(os.path.join(output_dir, 'test'))

        with pytest.raises(OutputDirExistsException):
            generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=False)

    # Test with exception for undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.undefined_var}}'))

        output_dir = os.path.join(tmpdir, 'output')
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'project_name': 'test_project'}, f)

    context = generate_context(context_file)
    assert context == {'cookiecutter': {'project_name': 'test_project'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context=default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    os.remove(context_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'key': 'default_value'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}
    os.remove(context_file)

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'key': 'default_value'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}
    os.remove(context_file)

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path("tests/mock_repo")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
            "_copy_without_render": ["*.bin"],
        }
    }

    # Test
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=True,
    )

    # Assert
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project").exists()
    assert (output_dir / "test_project" / "README.md").exists()
    assert (output_dir / "test_project" / "binary_file.bin").exists()

    # Cleanup
    rmtree(output_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context['test_context']['key'] == 'default_value'

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['test_context']['key'] == 'extra_value'

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent.json')


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_files():
    # Setup test data
    repo_dir = Path('test_template')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = Path('test_output')

    # Create a test template directory
    repo_dir.mkdir()
    (repo_dir / 'cookiecutter.json').write_text(json.dumps({'project_name': 'test_project'}))
    (repo_dir / '{{cookiecutter.project_name}}').mkdir()
    (repo_dir / '{{cookiecutter.project_name}}' / 'test_file.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    # Test generate_files function
    result = generate_files(repo_dir, context, output_dir)

    # Assertions
    assert result == str(output_dir / 'test_project')
    assert (output_dir / 'test_project' / 'test_file.txt').exists()
    assert (output_dir / 'test_project' / 'test_file.txt').read_text() == 'Hello, test_project!'

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context)
    assert context == {'test_context': {'key': 'default_value'}}

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    # Clean up
    os.remove(context_file)


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    test_dir = Path("test_generate_file")
    test_dir.mkdir(exist_ok=True)
    template_dir = test_dir / "template"
    template_dir.mkdir()
    project_dir = test_dir / "project"
    project_dir.mkdir()

    # Create a test template file
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello, {{ name }}!")

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "name": "World"
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test file generation
    generate_file(
        str(project_dir),
        "test.txt",
        context,
        env,
        skip_if_file_exists=False
    )

    # Verify the file was created and rendered correctly
    output_file = project_dir / "test.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

    # Cleanup
    rmtree(test_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context['test_context']['key'] == 'default_value'

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['test_context']['key'] == 'extra_value'

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    os.remove(context_file)


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context['test_context']['key'] == 'default_value'

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['test_context']['key'] == 'extra_value'

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    os.remove(context_file)


# LLM-generated content at query #35
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test list overwrite with valid choice
    context = {'choice_var': ['option1', 'option2', 'option3']}
    overwrite_context = {'choice_var': 'option2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choice_var'] == ['option2', 'option1', 'option3']

    # Test list overwrite with invalid choice
    context = {'choice_var': ['option1', 'option2']}
    overwrite_context = {'choice_var': 'invalid_option'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice overwrite with valid choices
    context = {'multi_choice': ['a', 'b', 'c']}
    overwrite_context = {'multi_choice': ['a', 'c']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['multi_choice'] == ['a', 'c']

    # Test multichoice overwrite with invalid choices
    context = {'multi_choice': ['a', 'b']}
    overwrite_context = {'multi_choice': ['a', 'd']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test boolean overwrite with valid string
    context = {'bool_var': True}
    overwrite_context = {'bool_var': 'y'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['bool_var'] is True

    # Test boolean overwrite with invalid string
    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'invalid'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test dictionary partial overwrite
    context = {'dict_var': {'key1': 'val1', 'key2': 'val2'}}
    overwrite_context = {'dict_var': {'key1': 'new_val1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['dict_var'] == {'key1': 'new_val1', 'key2': 'val2'}

    # Test new variable in dictionary
    context = {'dict_var': {'key1': 'val1'}}
    overwrite_context = {'dict_var': {'key1': 'new_val1', 'key2': 'val2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['dict_var'] == {'key1': 'new_val1', 'key2': 'val2'}

    # Test new variable at top level (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': 'new_value1'}


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    dirname = "test_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    result, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result.exists()
    assert created is True
    assert result.name == "test_dir"

    # Test directory creation with template rendering
    dirname = "{{ test }}"
    context = {"test": "rendered_dir"}
    output_dir = Path("/tmp")
    environment = Environment()

    result, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result.exists()
    assert created is True
    assert result.name == "rendered_dir"

    # Test existing directory without overwrite
    dirname = "existing_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    # Create the directory first
    existing_dir = output_dir / "existing_dir"
    existing_dir.mkdir(exist_ok=True)

    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)

    # Test existing directory with overwrite
    dirname = "existing_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    result, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result.exists()
    assert created is False
    assert result.name == "existing_dir"

    # Test empty directory name
    dirname = ""
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(dirname, context, output_dir, environment)


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test file generation
    generate_file(project_dir, infile, context, env)

    # Check if the file was created and rendered correctly
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic variable overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test adding new variable
    context = {'var1': 'value1'}
    overwrite_context = {'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'value1'
    assert context['var2'] == 'value2'

    # Test list variable with valid choice
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test list variable with invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice variable with valid choices
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test multichoice variable with invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test boolean variable with valid string
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean variable with invalid string
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test dictionary variable overwrite
    context = {'var1': {'subvar1': 'value1', 'subvar2': 'value2'}}
    overwrite_context = {'var1': {'subvar1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['subvar1'] == 'new_value1'
    assert context['var1']['subvar2'] == 'value2'

    # Test dictionary variable with new subvariable
    context = {'var1': {'subvar1': 'value1'}}
    overwrite_context = {'var1': {'subvar2': 'value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['subvar1'] == 'value1'
    assert context['var1']['subvar2'] == 'value2'

    # Test in_dictionary_variable flag
    context = {'var1': 'value1'}
    overwrite_context = {'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context['var1'] == 'value1'
    assert context['var2'] == 'value2'


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_files(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text(json.dumps({
        "project_name": "test_project",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    }))
    (repo_dir / "{{ cookiecutter.project_slug }}").mkdir()
    (repo_dir / "{{ cookiecutter.project_slug }}" / "test.txt").write_text("Hello {{ cookiecutter.project_name }}!")

    context = {"project_name": "Test Project"}
    output_dir = tmp_path / "output"

    # Mock the hook functions
    mocker.patch('cookiecutter.hooks.run_hook_from_repo_dir')

    # Test successful generation
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test.txt").exists()
    assert (output_dir / "test_project" / "test.txt").read_text() == "Hello Test Project!"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        skip_if_file_exists=True
    )

    # Test with undefined variable
    bad_context = {"bad_var": "test"}
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=repo_dir,
            context=bad_context,
            output_dir=tmp_path / "bad_output"
        )

    # Test with existing output directory
    with pytest.raises(OutputDirExistsException):
        generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=False
        )

    # Test with binary file
    binary_file = repo_dir / "{{ cookiecutter.project_slug }}" / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    context["_copy_without_render"] = ["*.bin"]

    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=tmp_path / "binary_output",
        overwrite_if_exists=True
    )

    assert (Path(result) / "binary.bin").exists()
    assert (Path(result) / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    # Test basic generation
    output_dir = tmp_path / "output"
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True
    )
    assert (output_dir / "test_project" / "test.txt").exists()
    assert (output_dir / "test_project" / "test.txt").read_text() == "Hello, test_project!"

    # Test with extra context
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True,
        extra_context={"project_name": "custom_project"}
    )
    assert (output_dir / "custom_project" / "test.txt").exists()
    assert (output_dir / "custom_project" / "test.txt").read_text() == "Hello, custom_project!"

    # Test skip_if_file_exists
    (output_dir / "custom_project" / "existing.txt").write_text("existing")
    (repo_dir / "{{cookiecutter.project_name}}" / "existing.txt").write_text("new")
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        extra_context={"project_name": "custom_project"}
    )
    assert (output_dir / "custom_project" / "existing.txt").read_text() == "existing"

    # Test hooks
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
    (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True,
        accept_hooks=True,
        extra_context={"project_name": "hook_project"}
    )
    assert (output_dir / "hook_project").exists()

    # Test error handling
    (repo_dir / "cookiecutter.json").write_text('{"invalid": "{{ undefined_var }}"}')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': 'new_value1', 'var2': 'value2'}

    # Test new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': 'new_value1'}

    # Test list choice variable
    context = {'choice_var': ['option1', 'option2', 'option3']}
    overwrite_context = {'choice_var': 'option2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'choice_var': ['option2', 'option1', 'option3']}

    # Test invalid choice variable
    context = {'choice_var': ['option1', 'option2']}
    overwrite_context = {'choice_var': 'invalid_option'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice variable
    context = {'multi_var': ['opt1', 'opt2', 'opt3']}
    overwrite_context = {'multi_var': ['opt2', 'opt3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'multi_var': ['opt2', 'opt3']}

    # Test invalid multichoice variable
    context = {'multi_var': ['opt1', 'opt2']}
    overwrite_context = {'multi_var': ['opt3']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test boolean variable
    context = {'bool_var': True}
    overwrite_context = {'bool_var': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'bool_var': True}

    # Test invalid boolean variable
    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'invalid'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test nested dictionary variable
    context = {'nested': {'key1': 'val1', 'key2': 'val2'}}
    overwrite_context = {'nested': {'key1': 'new_val1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'nested': {'key1': 'new_val1', 'key2': 'val2'}}

    # Test new nested dictionary variable
    context = {'nested': {'key1': 'val1'}}
    overwrite_context = {'nested': {'key1': 'new_val1', 'key2': 'val2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'nested': {'key1': 'new_val1', 'key2': 'val2'}}


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {
        'cookiecutter': {
            'name': 'test',
            'version': '1.0.0',
            'description': 'A test project'
        }
    }

    # Test context generation with default context
    default_context = {'name': 'default_name'}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['name'] == 'default_name'

    # Test context generation with extra context
    extra_context = {'version': '2.0.0'}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['version'] == '2.0.0'

    # Test context generation with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

    # Test context generation with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-data/nonexistent.json')

    # Test context generation with empty file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/empty.json')


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_file.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ name }}!')

    # Test normal file rendering
    context['name'] = 'World'
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'r', encoding='utf-8') as f:
        assert f.read() == 'Hello, World!'

    # Test binary file copy
    binary_file = 'test_binary.bin'
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, binary_file, context, env)
    with open(os.path.join(project_dir, binary_file), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

    # Test skip_if_file_exists
    context['name'] = 'New World'
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile), 'r', encoding='utf-8') as f:
        assert f.read() == 'Hello, World!'  # Should not be updated

    # Cleanup
    shutil.rmtree(project_dir)
    if os.path.exists(infile):
        os.remove(infile)
    if os.path.exists(binary_file):
        os.remove(binary_file)


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_file.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Assert
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, !'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_is_copy_only_path():
    # Test when path matches a pattern in context
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'static/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('static/image.png', context) is True

    # Test when path does not match any pattern
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('dynamic/image.png', context) is False

    # Test when context does not have the key
    empty_context = {}
    assert is_copy_only_path('file.txt', empty_context) is False

    # Test with nested paths
    assert is_copy_only_path('subdir/file.txt', context) is True
    assert is_copy_only_path('subdir/static/file.png', context) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    # Create test template file
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test
    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    if os.path.exists(outfile):
        os.remove(outfile)


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env
        )

        assert result_path == Path(tmpdir) / 'test_project'
        assert created is True
        assert result_path.exists()

    # Test directory overwriting
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        # First creation
        result_path, _ = render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=False
        )

        # Try to create again without overwrite
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                dirname, context, output_dir, env, overwrite_if_exists=False
            )

        # Now with overwrite
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=True
        )

        assert result_path == Path(tmpdir) / 'test_project'
        assert created is False  # Directory already existed
        assert result_path.exists()

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(dirname, context, output_dir, env)

    # Test undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.nonexistent_var }}'
        output_dir = Path(tmpdir)

        with pytest.raises(UndefinedVariableInTemplate):
            render_and_create_dir(dirname, context, output_dir, env)


# LLM-generated content at query #14
#--------------------------

```python
def test_is_copy_only_path():
    # Test when path matches a pattern in context
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/index.md', context) is True

    # Test when path does not match any pattern
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

    # Test when context does not have '_copy_without_render'
    context_no_pattern = {'cookiecutter': {}}
    assert is_copy_only_path('file.txt', context_no_pattern) is False

    # Test when context is empty
    empty_context = {}
    assert is_copy_only_path('file.txt', empty_context) is False


# LLM-generated content at query #15
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup test data
    project_dir = str(tmp_path / "test_project")
    os.makedirs(project_dir)

    # Create a test template file
    template_content = "Hello, {{ name }}!"
    template_file = tmp_path / "template.txt"
    template_file.write_text(template_content)

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        },
        'name': 'World'
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(tmp_path)))

    # Call the function
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env
    )

    # Check the result
    output_file = Path(project_dir) / "template.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

def test_generate_file_binary(tmp_path):
    # Setup test data
    project_dir = str(tmp_path / "test_project")
    os.makedirs(project_dir)

    # Create a binary test file
    binary_content = b'\x00\x01\x02\x03\x04'
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(binary_content)

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(tmp_path)))

    # Call the function
    generate_file(
        project_dir=project_dir,
        infile=str(binary_file),
        context=context,
        env=env
    )

    # Check the result
    output_file = Path(project_dir) / "binary.bin"
    assert output_file.exists()
    assert output_file.read_bytes() == binary_content

def test_generate_file_skip_if_exists(tmp_path):
    # Setup test data
    project_dir = str(tmp_path / "test_project")
    os.makedirs(project_dir)

    # Create a test template file
    template_content = "Hello, {{ name }}!"
    template_file = tmp_path / "template.txt"
    template_file.write_text(template_content)

    # Create existing output file
    output_file = Path(project_dir) / "template.txt"
    output_file.write_text("Existing content")

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        },
        'name': 'World'
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(tmp_path)))

    # Call the function with skip_if_file_exists=True
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env,
        skip_if_file_exists=True
    )

    # Check the result - file should not be overwritten
    assert output_file.read_text() == "Existing content"

def test_generate_file_empty_dirname(tmp_path):
    # Setup test data
    project_dir = str(tmp_path / "test_project")
    os.makedirs(project_dir)

    # Create a test template file with empty filename
    template_content = "Hello, {{ name }}!"
    template_file = tmp_path / "template.txt"
    template_file.write_text(template_content)

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        },
        'name': ''
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(tmp_path)))

    # Call the function - should handle empty filename gracefully
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env
    )

    # Check that no file was created
    output_file = Path(project_dir) / "template.txt"
    assert not output_file.exists()


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "tests/mock_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "tests/output"

    # Test successful generation
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

    # Test overwrite_if_exists
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

    # Test skip_if_file_exists
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    # Test accept_hooks
    generate_files(repo_dir, context, output_dir, accept_hooks=True)

    # Test keep_project_on_failure
    generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/test-fixtures/cookiecutter.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'test_project'

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-fixtures/invalid.json')

    # Test with missing file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-fixtures/missing.json')


# LLM-generated content at query #18
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list choice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test invalid list choice variable
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test multichoice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test invalid multichoice variable
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test dict variable
    context = {'var1': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'var1': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == {'key1': 'new_value1', 'key2': 'value2'}

    # Test boolean variable with 'yes'
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean variable with 'no'
    context = {'var1': False}
    overwrite_context = {'var1': 'no'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is False

    # Test invalid boolean variable
    context = {'var1': True}
    overwrite_context = {'var1': 'invalid_bool'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_bool" in str(e)

    # Test nested dict with new variable
    context = {'var1': {'key1': 'value1'}}
    overwrite_context = {'var1': {'key1': 'new_value1', 'key2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == {'key1': 'new_value1', 'key2': 'new_value2'}


# LLM-generated content at query #19
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test new variable in overwrite (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list choice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test invalid choice variable
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice variable
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test invalid multichoice variable
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test boolean variable
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test invalid boolean variable
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test nested dictionary variable
    context = {'var1': {'nested1': 'value1', 'nested2': 'value2'}}
    overwrite_context = {'var1': {'nested1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'value2'

    # Test new nested variable (should be added)
    context = {'var1': {'nested1': 'value1'}}
    overwrite_context = {'var1': {'nested1': 'new_value1', 'nested2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'new_value2'


# LLM-generated content at query #20
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test basic overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': 'new_value1', 'var2': 'value2'}

    # Test overwrite with new variable
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': 'new_value1'}

    # Test overwrite with list (choice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': ['choice2', 'choice1', 'choice3']}

    # Test overwrite with list (invalid choice)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'choice3'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test overwrite with list (multichoice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice2', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': ['choice2', 'choice3']}

    # Test overwrite with list (invalid multichoice)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test overwrite with dict
    context = {'var1': {'subvar1': 'subvalue1', 'subvar2': 'subvalue2'}}
    overwrite_context = {'var1': {'subvar1': 'new_subvalue1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': {'subvar1': 'new_subvalue1', 'subvar2': 'subvalue2'}}

    # Test overwrite with dict (new subvariable)
    context = {'var1': {'subvar1': 'subvalue1'}}
    overwrite_context = {'var1': {'subvar1': 'new_subvalue1', 'subvar2': 'subvalue2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': {'subvar1': 'new_subvalue1', 'subvar2': 'subvalue2'}}

    # Test overwrite with bool (valid)
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': True}

    # Test overwrite with bool (invalid)
    context = {'var1': True}
    overwrite_context = {'var1': 'invalid'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test overwrite with nested dict
    context = {'var1': {'subvar1': {'subsubvar1': 'value'}}}
    overwrite_context = {'var1': {'subvar1': {'subsubvar1': 'new_value'}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var1': {'subvar1': {'subsubvar1': 'new_value'}}}


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files
        result = generate_files(repo_dir, output_dir=output_dir)

        # Check result
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Hello test_project!'

    # Test with overwrite_if_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files first time
        generate_files(repo_dir, output_dir=output_dir)

        # Try to generate again with overwrite_if_exists=True
        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files first time
        generate_files(repo_dir, output_dir=output_dir)

        # Modify the generated file
        with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'w') as f:
            f.write('Modified content')

        # Try to generate again with skip_if_file_exists=True
        generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)

        # Check that the file was not overwritten
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Modified content'

    # Test with accept_hooks=False
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        # Create a pre-gen hook that would fail
        hooks_dir = os.path.join(repo_dir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('import sys; sys.exit(1)')

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files with accept_hooks=False should not run hooks
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=False)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

    # Test with keep_project_on_failure=True and failing hook
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        # Create a pre-gen hook that would fail
        hooks_dir = os.path.join(repo_dir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('import sys; sys.exit(1)')

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files with keep_project_on_failure=True should keep the project
        # even if hooks fail
        with pytest.raises(Exception):
            generate_files(repo_dir, output_dir=output_dir, keep_project_on_failure=True)

        # Check that the project directory still exists
        assert os.path.exists(os.path.join(output_dir, 'test_project'))

    # Test with context parameter
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        output_dir = os.path.join(temp_dir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'default_project'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files with custom context
        custom_context = {'project_name': 'custom_project'}
        result = generate_files(repo_dir, context=custom_context, output_dir=output_dir)

        # Check result
        assert os.path.exists(os.path.join(output_dir, 'custom_project', 'test.txt'))
        with open(os.path.join(output_dir, 'custom_project', 'test.txt')) as f:
            assert f.read() == 'Hello custom_project!'


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project'}, f)

        context = generate_context(context_file)
        assert context == {'cookiecutter': {'project_name': 'test_project'}}

    # Test with default_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project'}, f)

        default_context = {'project_name': 'default_project'}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project'}, f)

        extra_context = {'project_name': 'extra_project'}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with invalid JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{invalid json}')

        with pytest.raises(ContextDecodingException):
            generate_context(context_file)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    repo_dir = Path('tests/test-templates/basic-template')
    context = {'project_name': 'test-project', 'author': 'Test Author'}
    output_dir = Path('tests/test-outputs')
    project_dir = generate_files(repo_dir, context, output_dir)

    assert Path(project_dir).exists()
    assert Path(project_dir, 'README.md').exists()
    assert Path(project_dir, 'src').exists()
    assert Path(project_dir, 'src', 'test_project').exists()

    # Test with overwrite_if_exists
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(project_dir).exists()

    # Test with skip_if_file_exists
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(project_dir).exists()

    # Test with hooks
    repo_dir = Path('tests/test-templates/template-with-hooks')
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'hook_output.txt').exists()

    # Test with keep_project_on_failure
    repo_dir = Path('tests/test-templates/template-with-error')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(output_dir, 'template-with-error').exists()

    # Test with keep_project_on_failure=False
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=False)
    assert not Path(output_dir, 'template-with-error').exists()

    # Test with copy_only paths
    repo_dir = Path('tests/test-templates/template-with-copy-only')
    context = {'project_name': 'test-project'}
    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'copy_only.txt').exists()
    assert Path(project_dir, 'rendered.txt').exists()

    # Test with binary files
    repo_dir = Path('tests/test-templates/template-with-binary')
    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'binary_file.bin').exists()

    # Test with empty directory name
    repo_dir = Path('tests/test-templates/template-with-empty-dir')
    with pytest.raises(EmptyDirNameException):
        generate_files(repo_dir, context, output_dir)

    # Test with existing output directory
    repo_dir = Path('tests/test-templates/basic-template')
    project_dir = generate_files(repo_dir, context, output_dir)
    with pytest.raises(OutputDirExistsException):
        generate_files(repo_dir, context, output_dir, overwrite_if_exists=False)


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'key': 'default_value'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'key': 'extra_value'}}
    os.remove(context_file)

    # Test with invalid JSON file
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)

    # Create a test template file
    template_dir = tmp_path / "template"
    os.makedirs(template_dir)
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ name }}!")

    # Create context
    context = {
        'cookiecutter': {
            'name': 'World',
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    # Verify
    output_file = Path(project_dir) / "test.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

def test_generate_file_binary(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)

    # Create a binary file
    template_dir = tmp_path / "template"
    os.makedirs(template_dir)
    binary_file = template_dir / "test.bin"
    binary_content = b'\x00\x01\x02\x03'
    binary_file.write_bytes(binary_content)

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test
    generate_file(
        project_dir=project_dir,
        infile=str(binary_file),
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    # Verify
    output_file = Path(project_dir) / "test.bin"
    assert output_file.exists()
    assert output_file.read_bytes() == binary_content

def test_generate_file_skip_if_exists(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)

    # Create a test template file
    template_dir = tmp_path / "template"
    os.makedirs(template_dir)
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ name }}!")

    # Create existing output file
    output_file = Path(project_dir) / "test.txt"
    output_file.write_text("Existing content")

    # Create context
    context = {
        'cookiecutter': {
            'name': 'World',
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env,
        skip_if_file_exists=True
    )

    # Verify file wasn't modified
    assert output_file.read_text() == "Existing content"

def test_generate_file_undefined_variable(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)

    # Create a test template file with undefined variable
    template_dir = tmp_path / "template"
    os.makedirs(template_dir)
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ undefined_var }}!")

    # Create context without the required variable
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test and verify exception
    with pytest.raises(UndefinedError):
        generate_file(
            project_dir=project_dir,
            infile=str(template_file),
            context=context,
            env=env,
            skip_if_file_exists=False
        )


