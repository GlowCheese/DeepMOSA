####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    output_dir = tmp_path / "output"
    repo_dir.mkdir()
    output_dir.mkdir()

    # Create a simple template
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    # Test basic generation
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    # Verify output
    assert (output_dir / "test_project" / "file.txt").exists()
    assert (output_dir / "test_project" / "file.txt").read_text() == "Hello, test_project!"

    # Test with extra context
    result = generate_files(
        repo_dir=str(repo_dir),
        extra_context={"project_name": "another_project"},
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "another_project" / "file.txt").exists()
    assert (output_dir / "another_project" / "file.txt").read_text() == "Hello, another_project!"

    # Test with skip_if_file_exists
    (output_dir / "test_project" / "file.txt").write_text("Existing content")
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        skip_if_file_exists=True,
        overwrite_if_exists=True
    )

    assert (output_dir / "test_project" / "file.txt").read_text() == "Existing content"

    # Test with binary file
    binary_file = repo_dir / "{{cookiecutter.project_name}}" / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "test_project" / "binary.bin").exists()
    assert (output_dir / "test_project" / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'

    # Test with copy_without_render
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project", "_copy_without_render": ["*.md"]}')
    (repo_dir / "{{cookiecutter.project_name}}" / "readme.md").write_text("# {{cookiecutter.project_name}}")
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "test_project" / "readme.md").read_text() == "# {{cookiecutter.project_name}}"


# LLM-generated content at query #2
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

    # Test list overwrite with valid choice
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test list overwrite with invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test list overwrite with valid multichoice
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test list overwrite with invalid multichoice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test dict overwrite
    context = {'var1': {'subvar1': 'subvalue1', 'subvar2': 'subvalue2'}}
    overwrite_context = {'var1': {'subvar1': 'new_subvalue1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['subvar1'] == 'new_subvalue1'
    assert context['var1']['subvar2'] == 'subvalue2'

    # Test dict with new variable
    context = {'var1': {'subvar1': 'subvalue1'}}
    overwrite_context = {'var1': {'subvar1': 'new_subvalue1', 'subvar2': 'new_subvalue2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['subvar1'] == 'new_subvalue1'
    assert context['var1']['subvar2'] == 'new_subvalue2'

    # Test boolean overwrite with valid string
    context = {'var1': True}
    overwrite_context = {'var1': 'y'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean overwrite with invalid string
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test simple overwrite
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'


# LLM-generated content at query #3
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

    # Test list overwrite with valid choice
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test list overwrite with invalid choice (should raise ValueError)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test multichoice overwrite with valid subset
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test multichoice overwrite with invalid subset (should raise ValueError)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test dictionary overwrite
    context = {'var1': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'var1': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'value2'

    # Test boolean overwrite with valid string
    context = {'var1': True}
    overwrite_context = {'var1': 'y'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean overwrite with invalid string (should raise ValueError)
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test nested dictionary with new variable
    context = {'var1': {'key1': 'value1'}}
    overwrite_context = {'var1': {'key1': 'new_value1', 'key2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'new_value2'


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test with existing directory (should raise exception)
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test with overwrite_if_exists=True
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created  # Directory already existed
        assert dir_to_create.name == 'test_project'

    # Test with empty directory name (should raise exception)
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test with template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'author': 'test_author'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}_{{cookiecutter.author}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'my_project_test_author'


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_files(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test_file.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(repo_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(repo_dir))))
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Test generate_files
    output_dir = tmp_path / "output"
    result = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    # Assertions
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test_file.txt").exists()
    assert (output_dir / "test_project" / "test_file.txt").read_text() == "Hello, test_project!"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True
    )
    assert (output_dir / "test_project" / "test_file.txt").exists()

    # Test with accept_hooks=False
    generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path / "output_no_hooks",
        accept_hooks=False
    )


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json')
        assert context == {'test': {'key': 'value'}}

    # Test with default_context
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json', default_context={'key': 'new_value'})
        assert context == {'test': {'key': 'new_value'}}

    # Test with extra_context
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json', extra_context={'key': 'extra_value'})
        assert context == {'test': {'key': 'extra_value'}}

    # Test with invalid JSON
    with patch('builtins.open', mock_open(read_data='invalid json')) as mock_file:
        with pytest.raises(ContextDecodingException):
            generate_context('test.json')

    # Test with non-existent file
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            generate_context('nonexistent.json')


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mocks/cookiecutter.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'My Project'
    assert context['cookiecutter']['project_slug'] == 'my_project'

    # Test with default context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra context
    extra_context = {'project_slug': 'extra_slug'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_slug'] == 'extra_slug'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mocks/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mocks/nonexistent.json')


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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

    # Test when context does not have '_copy_without_render'
    context_no_pattern = {'cookiecutter': {}}
    assert is_copy_only_path('file.txt', context_no_pattern) is False

    # Test when context is missing 'cookiecutter' key
    context_missing_key = {}
    assert is_copy_only_path('file.txt', context_missing_key) is False


# LLM-generated content at query #10
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
    overwrite_context = {'var1': 'new_value1', 'var2': 'value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test list variable with valid choice
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test list variable with invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test multichoice variable with valid choices
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice1', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice1', 'choice3']

    # Test multichoice variable with invalid choice
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test boolean variable with valid string
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test boolean variable with invalid string
    context = {'var1': False}
    overwrite_context = {'var1': 'invalid_bool'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_bool" in str(e)

    # Test nested dictionary overwrite
    context = {'var1': {'nested1': 'value1', 'nested2': 'value2'}}
    overwrite_context = {'var1': {'nested1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'value2'

    # Test nested dictionary with new key
    context = {'var1': {'nested1': 'value1'}}
    overwrite_context = {'var1': {'nested1': 'new_value1', 'nested2': 'value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested1'] == 'new_value1'
    assert context['var1']['nested2'] == 'value2'


# LLM-generated content at query #11
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
    shutil.rmtree(project_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    infile = "test.txt"
    context = {"cookiecutter": {"_copy_without_render": [], "_new_lines": "\n"}}
    env = Environment(loader=FileSystemLoader("."))

    # Create a test file
    test_file = Path(infile)
    test_file.write_text("Hello, {{ name }}!")

    # Test binary file
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"binary data")
    context_binary = context.copy()
    context_binary["cookiecutter"]["_copy_without_render"] = ["*.bin"]

    # Test
    # Test 1: Normal file rendering
    context["name"] = "World"
    generate_file(str(project_dir), infile, context, env)
    assert (project_dir / infile).read_text() == "Hello, World!"

    # Test 2: Binary file copying
    generate_file(str(project_dir), str(binary_file), context_binary, env)
    assert (project_dir / binary_file.name).read_bytes() == b"binary data"

    # Test 3: Skip if file exists
    generate_file(str(project_dir), infile, context, env, skip_if_file_exists=True)
    assert (project_dir / infile).read_text() == "Hello, World!"

    # Test 4: Empty directory name
    empty_dir = tmp_path / ""
    with pytest.raises(EmptyDirNameException):
        generate_file(str(empty_dir), infile, context, env)

    # Test 5: Undefined variable in template
    undefined_context = {"cookiecutter": {"_copy_without_render": []}}
    with pytest.raises(UndefinedVariableInTemplate):
        generate_file(str(project_dir), infile, undefined_context, env)

    # Cleanup
    test_file.unlink()


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
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

    # Test binary file copy
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    generate_file(project_dir, str(binary_file), context, env)
    assert (Path(project_dir) / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'

    # Test text file rendering
    generate_file(project_dir, str(template_file), context, env)
    output_file = Path(project_dir) / "template.txt"
    assert output_file.read_text() == "Hello, World!"

    # Test skip_if_file_exists
    generate_file(project_dir, str(template_file), context, env, skip_if_file_exists=True)
    assert output_file.read_text() == "Hello, World!"  # Should not change

    # Test with different newline
    context['cookiecutter']['_new_lines'] = '\r\n'
    template_file.write_text("Line1\nLine2")
    generate_file(project_dir, str(template_file), context, env)
    assert (Path(project_dir) / "template.txt").read_text() == "Line1\r\nLine2"

    # Test with template syntax error
    bad_template = tmp_path / "bad.txt"
    bad_template.write_text("{% if %}")
    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, str(bad_template), context, env)

    # Test with undefined variable
    bad_context = {'cookiecutter': {'_copy_without_render': []}}
    with pytest.raises(UndefinedError):
        generate_file(project_dir, str(template_file), bad_context, env)


# LLM-generated content at query #14
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
    shutil.rmtree(project_dir)


# LLM-generated content at query #15
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with context override
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            context={"cookiecutter": {"project_name": "override"}},
            overwrite_if_exists=True
        )

        assert (output_dir / "override" / "file.txt").exists()
        assert (output_dir / "override" / "file.txt").read_text() == "override"

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("new content")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        assert (output_dir / "test" / "file.txt").read_text() == "new content"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True,
            overwrite_if_exists=True
        )

        assert (output_dir / "test").exists()

    # Test error handling for undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir,
                overwrite_if_exists=True
            )


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))

        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with overwrite_if_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        output_dir = Path(tmpdir) / "output"
        (output_dir / "test").mkdir()

        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True
        )
        assert (output_dir / "test").exists()

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        output_dir = Path(tmpdir) / "output"
        (output_dir / "test").mkdir()
        (output_dir / "test" / "file.txt").write_text("existing")

        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            skip_if_file_exists=True
        )
        assert (output_dir / "test" / "file.txt").read_text() == "existing"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post')")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            accept_hooks=True
        )
        assert (output_dir / "test").exists()

    # Test with binary file
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        binary_file = repo_dir / "{{cookiecutter.project_name}}" / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))

        assert (output_dir / "test" / "binary.bin").exists()
        assert (output_dir / "test" / "binary.bin").read_bytes() == b"\x00\x01\x02"

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text(
            '{"project_name": "test", "_copy_without_render": ["*.md"]}'
        )
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "readme.md").write_text("# {{cookiecutter.project_name}}")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))

        assert (output_dir / "test" / "readme.md").read_text() == "# {{cookiecutter.project_name}}"

    # Test with undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        output_dir = Path(tmpdir) / "output"
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(str(repo_dir), output_dir=str(output_dir))

    # Test with existing output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        output_dir = Path(tmpdir) / "output"
        (output_dir / "test").mkdir()

        with pytest.raises(OutputDirExistsException):
            generate_files(str(repo_dir), output_dir=str(output_dir))


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello")

    # Test basic generation
    project_dir = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path,
        overwrite_if_exists=True
    )

    assert (tmp_path / "test_project" / "test.txt").exists()
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Hello"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path,
        skip_if_file_exists=True
    )

    # Test with hooks
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
    (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

    project_dir = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project2"}},
        output_dir=tmp_path,
        accept_hooks=True
    )

    assert (tmp_path / "test_project2").exists()

    # Test error cases
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context={"cookiecutter": {"project_name": "{{undefined_var}}"}},
            output_dir=tmp_path
        )

    # Test with binary file
    binary_file = repo_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02')
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project3", "_copy_without_render": ["*.bin"]}')

    project_dir = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project3"}},
        output_dir=tmp_path
    )

    assert (tmp_path / "test_project3" / "binary.bin").exists()


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': [],
            '_new_lines': '\n'
        }
    }
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.project_name }}!')

    # Execute
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test_project!'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"

    # Create a test template directory
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "{{cookiecutter.project_name}}"), exist_ok=True)

    # Create a test file in the template
    test_file_path = os.path.join(repo_dir, "{{cookiecutter.project_name}}", "test.txt")
    with open(test_file_path, "w") as f:
        f.write("Hello, {{cookiecutter.project_name}}!")

    # Execute
    result = generate_files(repo_dir, context, output_dir)

    # Verify
    expected_output_dir = os.path.join(output_dir, "test_project")
    assert result == expected_output_dir
    assert os.path.exists(expected_output_dir)

    # Check if the file was generated correctly
    generated_file_path = os.path.join(expected_output_dir, "test.txt")
    assert os.path.exists(generated_file_path)
    with open(generated_file_path, "r") as f:
        content = f.read()
    assert content == "Hello, test_project!"

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory creation with existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        # Should raise exception when directory exists
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                context,
                tmpdir,
                env,
                overwrite_if_exists=False
            )

        # Should overwrite when flag is True
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created  # Because it existed before

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                context,
                tmpdir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            context,
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'my_project-1.0'


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_files(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    output_dir = tmp_path / "output"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(repo_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(repo_dir))))
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=True
    )

    # Assert
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test.txt").exists()
    assert (output_dir / "test_project" / "test.txt").read_text() == "Hello, test_project!"

def test_generate_files_with_hooks(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    output_dir = tmp_path / "output"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(repo_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(repo_dir))))
    mock_hook = mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=True
    )

    # Assert
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test.txt").exists()
    assert (output_dir / "test_project" / "test.txt").read_text() == "Hello, test_project!"
    assert mock_hook.call_count == 2  # pre and post hooks

def test_generate_files_skip_existing(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    output_dir = tmp_path / "output"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(repo_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(repo_dir))))
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Create existing file
    (output_dir / "test_project").mkdir()
    (output_dir / "test_project" / "test.txt").write_text("Existing content")

    # Execute
    result = generate_files(
        repo_dir=str(repo_dir),
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=False,
        skip_if_file_exists=True,
        accept_hooks=False,
        keep_project_on_failure=True
    )

    # Assert
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project" / "test.txt").read_text() == "Existing content"

def test_generate_files_undefined_variable(tmp_path, mocker):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.undefined_var}}!")

    output_dir = tmp_path / "output"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Mock functions
    mocker.patch('cookiecutter.generate.find_template', return_value=str(repo_dir))
    mocker.patch('cookiecutter.generate.create_env_with_context', return_value=Environment(loader=FileSystemLoader(str(repo_dir))))
    mocker.patch('cookiecutter.generate.run_hook_from_repo_dir')

    # Execute and assert exception
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=True,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=True
        )


# LLM-generated content at query #22
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"test_var": "test_value"}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = False

    # Test when directory name is empty
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment, overwrite_if_exists)

    # Test when directory does not exist
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
    assert dir_to_create.exists()
    assert created is True

    # Test when directory exists and overwrite_if_exists is False
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)

    # Test when directory exists and overwrite_if_exists is True
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, True)
    assert dir_to_create.exists()
    assert created is False

    # Cleanup
    if dir_to_create.exists():
        shutil.rmtree(dir_to_create)


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    template_dir = str(tmp_path / "template")
    os.makedirs(template_dir)

    # Create a test file in the template directory
    test_file = os.path.join(template_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, {{ name }}!")

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "name": "World"
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Test
    with work_in(template_dir):
        generate_file(
            project_dir=project_dir,
            infile="test.txt",
            context=context,
            env=env,
            skip_if_file_exists=False
        )

    # Assert
    output_file = os.path.join(project_dir, "test.txt")
    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello, World!"

def test_generate_file_binary(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    template_dir = str(tmp_path / "template")
    os.makedirs(template_dir)

    # Create a binary test file in the template directory
    test_file = os.path.join(template_dir, "test.bin")
    with open(test_file, "wb") as f:
        f.write(b"\x00\x01\x02\x03")

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Test
    with work_in(template_dir):
        generate_file(
            project_dir=project_dir,
            infile="test.bin",
            context=context,
            env=env,
            skip_if_file_exists=False
        )

    # Assert
    output_file = os.path.join(project_dir, "test.bin")
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_skip_if_exists(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    template_dir = str(tmp_path / "template")
    os.makedirs(template_dir)

    # Create a test file in the template directory
    test_file = os.path.join(template_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, {{ name }}!")

    # Create the output file before calling generate_file
    output_file = os.path.join(project_dir, "test.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Existing content")

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "name": "World"
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Test
    with work_in(template_dir):
        generate_file(
            project_dir=project_dir,
            infile="test.txt",
            context=context,
            env=env,
            skip_if_file_exists=True
        )

    # Assert
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Existing content"

def test_generate_file_undefined_variable(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)
    template_dir = str(tmp_path / "template")
    os.makedirs(template_dir)

    # Create a test file in the template directory with an undefined variable
    test_file = os.path.join(template_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, {{ undefined_variable }}!")

    # Create context without the undefined variable
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Test and Assert
    with work_in(template_dir):
        with pytest.raises(UndefinedVariableInTemplate):
            generate_file(
                project_dir=project_dir,
                infile="test.txt",
                context=context,
                env=env,
                skip_if_file_exists=False
            )


# LLM-generated content at query #24
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
    shutil.rmtree(project_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ name }}!')

    # Test rendering
    context['name'] = 'World'
    generate_file(project_dir, infile, context, env)

    # Verify the output file
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, World!'

    # Test binary file copy
    binary_file = 'test_binary.bin'
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, binary_file, context, env)
    binary_outfile = os.path.join(project_dir, binary_file)
    assert os.path.exists(binary_outfile)
    with open(binary_outfile, 'rb') as f:
        binary_content = f.read()
    assert binary_content == b'\x00\x01\x02\x03'

    # Cleanup
    shutil.rmtree(project_dir)
    os.remove(infile)
    os.remove(binary_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
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
        assert context == {'cookiecutter': {'project_name': 'default_project'}}

    # Test with extra_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project'}, f)

        extra_context = {'project_name': 'extra_project'}
        context = generate_context(context_file, extra_context=extra_context)
        assert context == {'cookiecutter': {'project_name': 'extra_project'}}

    # Test with invalid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('invalid json')

        with pytest.raises(ContextDecodingException):
            generate_context(context_file)

    # Test with non-existent JSON file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')


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
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create = Path(tmpdir) / 'test_project'
        dir_to_create.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create = Path(tmpdir) / 'test_project'
        dir_to_create.mkdir()

        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created
        assert dir_to_create.name == 'test_project'

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'my_project-1.0'


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (template_dir / "{{cookiecutter.project_name}}").mkdir()
    (template_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

    # Test basic file generation
    result = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True
    )
    assert (tmp_path / "test_project" / "file.txt").exists()
    assert (tmp_path / "test_project" / "file.txt").read_text() == "content"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        skip_if_file_exists=True
    )
    assert (tmp_path / "test_project" / "file.txt").exists()

    # Test with overwrite_if_exists
    generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True
    )
    assert (tmp_path / "test_project" / "file.txt").exists()

    # Test with hooks
    (template_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
    (template_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")
    result = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        accept_hooks=True,
        overwrite_if_exists=True
    )
    assert (tmp_path / "test_project" / "file.txt").exists()

    # Test with keep_project_on_failure
    (template_dir / "hooks" / "pre_gen_project.py").write_text("raise Exception('test')")
    with pytest.raises(Exception):
        generate_files(
            repo_dir=str(template_dir),
            context={"cookiecutter": {"project_name": "test_project"}},
            output_dir=str(tmp_path),
            accept_hooks=True,
            keep_project_on_failure=True
        )
    assert (tmp_path / "test_project").exists()

    # Test with binary file
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02")
    shutil.copy(binary_file, template_dir / "binary.bin")
    result = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True
    )
    assert (tmp_path / "test_project" / "binary.bin").exists()
    assert (tmp_path / "test_project" / "binary.bin").read_bytes() == b"\x00\x01\x02"


# LLM-generated content at query #30
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert created

        # Create again with overwrite
        dir_to_create2, created2 = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create2.exists()
        assert not created2  # Should not be created again

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test'}
        env = Environment()
        with pytest.raises(UndefinedVariableInTemplate):
            render_and_create_dir(
                '{{cookiecutter.nonexistent}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test existing directory without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created

        # Try to create again without overwrite
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )


# LLM-generated content at query #31
#--------------------------

```python
def test_render_and_create_dir():
    # Test normal directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)

        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            output_dir,
            env
        )

        assert created
        assert result_dir.exists()
        assert result_dir.name == 'test_project'

    # Test directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)

        # Create the directory first
        test_dir = Path(tmpdir) / 'test_project'
        test_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                output_dir,
                env,
                overwrite_if_exists=False
            )

    # Test directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)

        # Create the directory first
        test_dir = Path(tmpdir) / 'test_project'
        test_dir.mkdir()

        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            output_dir,
            env,
            overwrite_if_exists=True
        )

        assert not created
        assert result_dir.exists()
        assert result_dir.name == 'test_project'

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        output_dir = Path(tmpdir)

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                output_dir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'author': 'test_author'}
        env = Environment()
        output_dir = Path(tmpdir)

        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}_{{cookiecutter.author}}',
            {'cookiecutter': context},
            output_dir,
            env
        )

        assert created
        assert result_dir.exists()
        assert result_dir.name == 'my_project_test_author'


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup test data
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    template_dir = tmp_path / "template"
    template_dir.mkdir()

    # Create a test file in the template directory
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
    with work_in(str(template_dir)):
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

    # Test binary file handling
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")

    with work_in(str(template_dir)):
        generate_file(
            str(project_dir),
            "binary.bin",
            context,
            env,
            skip_if_file_exists=False
        )

    output_binary = project_dir / "binary.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == b"\x00\x01\x02\x03"

    # Test skip_if_file_exists
    with work_in(str(template_dir)):
        generate_file(
            str(project_dir),
            "test.txt",
            context,
            env,
            skip_if_file_exists=True
        )

    # File should still have original content
    assert output_file.read_text() == "Hello, World!"

    # Test template syntax error
    bad_template = template_dir / "bad.txt"
    bad_template.write_text("{% if %}")

    with work_in(str(template_dir)):
        with pytest.raises(TemplateSyntaxError):
            generate_file(
                str(project_dir),
                "bad.txt",
                context,
                env,
                skip_if_file_exists=False
            )

    # Test undefined variable in template
    bad_context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    with work_in(str(template_dir)):
        with pytest.raises(UndefinedError):
            generate_file(
                str(project_dir),
                "test.txt",
                bad_context,
                env,
                skip_if_file_exists=False
            )


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # First generation
        generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test").exists()

        # Second generation with overwrite
        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True
        )
        assert (output_dir / "test").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("original")

        # First generation
        generate_files(str(repo_dir), output_dir=str(output_dir))

        # Modify the output file
        (output_dir / "test" / "file.txt").write_text("modified")

        # Second generation with skip
        generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            skip_if_file_exists=True
        )

        # File should keep its modified content
        assert (output_dir / "test" / "file.txt").read_text() == "modified"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with hooks
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text(
            "import os\nwith open('hook_test.txt', 'w') as f: f.write('pre')"
        )
        (repo_dir / "hooks" / "post_gen_project.py").write_text(
            "import os\nwith open('hook_test.txt', 'a') as f: f.write('post')"
        )

        result = generate_files(str(repo_dir), output_dir=str(output_dir), accept_hooks=True)
        assert (output_dir / "test" / "hook_test.txt").read_text() == "prepost"

    # Test with context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "default"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        # Generate with custom context
        context = {"project_name": "custom"}
        result = generate_files(str(repo_dir), context=context, output_dir=str(output_dir))
        assert (output_dir / "custom" / "file.txt").read_text() == "custom"

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "test",
            "_copy_without_render": ["*.bin"]
        }))
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.bin").write_bytes(b"binary content")

        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "file.bin").read_bytes() == b"binary content"

    # Test error cases
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with undefined variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.undefined_var}}")

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(str(repo_dir), output_dir=str(output_dir))

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template and existing output directory
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        output_dir.mkdir()
        (output_dir / "test").mkdir()

        with pytest.raises(OutputDirExistsException):
            generate_files(str(repo_dir), output_dir=str(output_dir))


# LLM-generated content at query #35
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


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mock-template/cookiecutter.json')
    assert context['cookiecutter']['project_name'] == 'My Project'
    assert context['cookiecutter']['project_slug'] == 'my_project'

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/mock-template/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_slug': 'extra_slug'}
    context = generate_context(
        'tests/mock-template/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_slug'] == 'extra_slug'

    # Test with both default and extra context
    context = generate_context(
        'tests/mock-template/cookiecutter.json',
        default_context=default_context,
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'
    assert context['cookiecutter']['project_slug'] == 'extra_slug'

    # Test invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-json.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/non-existent.json')


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup test environment
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    template_dir = tmp_path / "template"
    template_dir.mkdir()

    # Create a test file in the template directory
    test_file = template_dir / "test.txt"
    test_file.write_text("Hello, {{ name }}!")

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': None
        },
        'name': 'World'
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test file generation
    with work_in(str(template_dir)):
        generate_file(
            str(project_dir),
            "test.txt",
            context,
            env,
            skip_if_file_exists=False
        )

    # Check if file was created and rendered correctly
    output_file = project_dir / "test.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

    # Test binary file handling
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')

    with work_in(str(template_dir)):
        generate_file(
            str(project_dir),
            "binary.bin",
            context,
            env,
            skip_if_file_exists=False
        )

    output_binary = project_dir / "binary.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x00\x01\x02\x03'

    # Test skip_if_file_exists
    with work_in(str(template_dir)):
        generate_file(
            str(project_dir),
            "test.txt",
            context,
            env,
            skip_if_file_exists=True
        )

    # File should not be overwritten
    assert output_file.read_text() == "Hello, World!"

    # Test template syntax error
    bad_template = template_dir / "bad.txt"
    bad_template.write_text("Hello, {% if %}!")

    with work_in(str(template_dir)):
        with pytest.raises(TemplateSyntaxError):
            generate_file(
                str(project_dir),
                "bad.txt",
                context,
                env,
                skip_if_file_exists=False
            )

    # Test undefined variable in template
    bad_context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': None
        }
    }

    with work_in(str(template_dir)):
        with pytest.raises(UndefinedVariableInTemplate):
            generate_file(
                str(project_dir),
                "test.txt",
                bad_context,
                env,
                skip_if_file_exists=False
            )


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path("tests/mocks/valid-template")
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = Path("tests/output")

    # Test normal generation
    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir).exists()
    assert Path(project_dir, "test_project").exists()

    # Test overwrite_if_exists
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(project_dir).exists()

    # Test skip_if_file_exists
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(project_dir).exists()

    # Test with hooks
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(project_dir).exists()

    # Test keep_project_on_failure
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(project_dir).exists()

    # Cleanup
    rmtree(output_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mock_template/cookiecutter.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'My Project'

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/mock_template/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(
        'tests/mock_template/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mock_template/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mock_template/nonexistent.json')

    # Test with empty context file
    context = generate_context('tests/mock_template/empty.json')
    assert context == OrderedDict([])

    # Test with nested context
    context = generate_context('tests/mock_template/nested.json')
    assert context['cookiecutter']['nested']['key'] == 'value'


# LLM-generated content at query #40
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    context = {'project_name': 'test_project'}
    environment = Environment()
    output_dir = Path('/tmp/test_output')

    # Test normal directory creation
    rendered_dir, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        environment
    )
    assert rendered_dir == output_dir / 'test_project'
    assert created is True
    assert rendered_dir.exists()

    # Test with existing directory and overwrite=False
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            output_dir,
            environment,
            overwrite_if_exists=False
        )

    # Test with existing directory and overwrite=True
    rendered_dir, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        environment,
        overwrite_if_exists=True
    )
    assert rendered_dir == output_dir / 'test_project'
    assert created is False
    assert rendered_dir.exists()

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, environment)

    # Cleanup
    if rendered_dir.exists():
        rmtree(rendered_dir)


# LLM-generated content at query #41
#--------------------------

```python
def test_generate_context():
    # Test with valid context file
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


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_context():
    # Test with a simple context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'project_name': 'test_project'}, f)

    context = generate_context(context_file)
    assert context == {'test_context': {'project_name': 'test_project'}}
    os.remove(context_file)

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context=default_context)
    assert context == {'test_context': {'project_name': 'default_project'}}
    os.remove(context_file)

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'test_context': {'project_name': 'extra_project'}}
    os.remove(context_file)

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')
    with pytest.raises(ContextDecodingException):
        generate_context(context_file)
    os.remove(context_file)


# LLM-generated content at query #43
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = "/tmp/test_project"
    infile = "test_template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment(loader=FileSystemLoader("."))

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
    if os.path.exists(outfile):
        os.remove(outfile)


# LLM-generated content at query #44
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ name }}!')

    # Test normal file generation
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
    os.remove(infile)
    os.remove(binary_file)


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        # Generate files
        result = generate_files(
            repo_dir,
            context={"cookiecutter": {"project_name": "test"}},
            output_dir=output_dir
        )

        # Verify output
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # First generation
        generate_files(repo_dir, output_dir=output_dir)

        # Second generation with overwrite
        result = generate_files(
            repo_dir,
            output_dir=output_dir,
            overwrite_if_exists=True
        )
        assert result == str(output_dir / "test")

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("original")

        # First generation
        generate_files(repo_dir, output_dir=output_dir)

        # Modify the output file
        (output_dir / "test" / "file.txt").write_text("modified")

        # Second generation with skip
        generate_files(
            repo_dir,
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        # Verify file wasn't overwritten
        assert (output_dir / "test" / "file.txt").read_text() == "modified"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with hooks
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

        # Generate with hooks
        result = generate_files(
            repo_dir,
            output_dir=output_dir,
            accept_hooks=True
        )
        assert result == str(output_dir / "test")

    # Test with undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with undefined variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        # Should raise exception
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with copy_without_render
        repo_dir.mkdir()
        context = {
            "cookiecutter": {
                "project_name": "test",
                "_copy_without_render": ["*.bin"]
            }
        }
        (repo_dir / "cookiecutter.json").write_text(json.dumps(context["cookiecutter"]))
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.bin").write_text("binary")

        # Generate files
        result = generate_files(
            repo_dir,
            context=context,
            output_dir=output_dir
        )

        # Verify binary file was copied without rendering
        assert (output_dir / "test" / "file.bin").exists()
        assert (output_dir / "test" / "file.bin").read_text() == "binary"


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_files():
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple template directory structure
        template_dir = Path(tmpdir) / "template"
        template_dir.mkdir()

        # Create a cookiecutter.json file
        context_file = template_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "test_project", "author": "test_author"}')

        # Create a simple template file
        template_file = template_dir / "{{cookiecutter.project_name}}"
        template_file.mkdir()
        (template_file / "test.txt").write_text("Hello, {{cookiecutter.author}}!")

        # Create a copy-only file
        copy_file = template_dir / "copy_only.txt"
        copy_file.write_text("This file should be copied without rendering.")

        # Create a directory to copy without rendering
        copy_dir = template_dir / "copy_dir"
        copy_dir.mkdir()
        (copy_dir / "copy_file.txt").write_text("This file is in a copy-only directory.")

        # Create a context with _copy_without_render
        context = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author",
                "_copy_without_render": ["copy_only.txt", "copy_dir"]
            }
        }

        # Create an output directory
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Test
        result = generate_files(
            repo_dir=str(template_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )

        # Assertions
        assert result == str(output_dir / "test_project")

        # Check if the rendered file exists and has correct content
        rendered_file = Path(result) / "test.txt"
        assert rendered_file.exists()
        assert rendered_file.read_text() == "Hello, test_author!"

        # Check if the copy-only file exists and has correct content
        copy_only_file = Path(result) / "copy_only.txt"
        assert copy_only_file.exists()
        assert copy_only_file.read_text() == "This file should be copied without rendering."

        # Check if the copy-only directory exists and has correct content
        copy_only_dir = Path(result) / "copy_dir"
        assert copy_only_dir.exists()
        copy_file_in_dir = copy_only_dir / "copy_file.txt"
        assert copy_file_in_dir.exists()
        assert copy_file_in_dir.read_text() == "This file is in a copy-only directory."

        # Test with skip_if_file_exists=True
        # Create a file that should be skipped
        skip_file = Path(result) / "skip.txt"
        skip_file.write_text("This file should be skipped.")

        # Create a template file that would generate the skip file
        skip_template = template_file / "skip.txt"
        skip_template.write_text("This file should be skipped.")

        # Generate files again with skip_if_file_exists=True
        result = generate_files(
            repo_dir=str(template_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            skip_if_file_exists=True,
            accept_hooks=False,
            keep_project_on_failure=False
        )

        # Assert that the skip file was not overwritten
        assert skip_file.read_text() == "This file should be skipped."

        # Test with overwrite_if_exists=False
        # Create a directory that should not be overwritten
        existing_dir = Path(result) / "existing_dir"
        existing_dir.mkdir()

        # Create a template directory that would generate the existing_dir
        existing_template_dir = template_file / "existing_dir"
        existing_template_dir.mkdir()

        # Generate files again with overwrite_if_exists=False
        try:
            generate_files(
                repo_dir=str(template_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                skip_if_file_exists=False,
                accept_hooks=False,
                keep_project_on_failure=False
            )
            assert False, "Expected OutputDirExistsException"
        except OutputDirExistsException:
            pass

        # Test with undefined variable in template
        # Create a template file with an undefined variable
        undefined_template = template_file / "undefined.txt"
        undefined_template.write_text("Hello, {{cookiecutter.undefined_variable}}!")

        # Generate files again with undefined variable
        try:
            generate_files(
                repo_dir=str(template_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                skip_if_file_exists=False,
                accept_hooks=False,
                keep_project_on_failure=False
            )
            assert False, "Expected UndefinedVariableInTemplate"
        except UndefinedVariableInTemplate:
            pass


# LLM-generated content at query #47
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': ['*.bin'],
            '_new_lines': '\n'
        }
    }
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.project_name }}!')

    # Test text file rendering
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test_project!'

    # Test binary file copying
    binary_file = 'test.bin'
    with open(binary_file, 'wb') as f:
        f.write(b'test binary content')
    generate_file(project_dir, binary_file, context, env)
    binary_outfile = os.path.join(project_dir, binary_file)
    assert os.path.exists(binary_outfile)
    with open(binary_outfile, 'rb') as f:
        binary_content = f.read()
    assert binary_content == b'test binary content'

    # Test skip_if_file_exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, 'r', encoding='utf-8') as f:
        content_after_skip = f.read()
    assert content_after_skip == 'Hello, test_project!'

    # Cleanup
    os.remove(outfile)
    os.remove(binary_outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path("tests/fake-repo-pre")
    output_dir = Path("tests/output")
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
            "_copy_without_render": ["*.md", "LICENSE"],
        }
    }

    # Execute
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=True,
    )

    # Verify
    assert Path(result).exists()
    assert Path(result, "test_project").exists()
    assert Path(result, "test_project", "README.md").exists()
    assert Path(result, "test_project", "LICENSE").exists()
    assert Path(result, "test_project", "setup.py").exists()

    # Check content rendering
    with open(Path(result, "test_project", "setup.py"), "r") as f:
        content = f.read()
        assert "test_project" in content


# LLM-generated content at query #49
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(repo_dir, output_dir=output_dir)
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with context override
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        context = {"project_name": "override"}
        result = generate_files(repo_dir, context=context, output_dir=output_dir)
        assert (output_dir / "override" / "file.txt").exists()
        assert (output_dir / "override" / "file.txt").read_text() == "override"

    # Test with existing output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "existing"

        repo_dir.mkdir()
        output_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Should raise OutputDirExistsException
        with pytest.raises(OutputDirExistsException):
            generate_files(repo_dir, output_dir=output_dir)

        # Should work with overwrite_if_exists=True
        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert (output_dir / "test").exists()

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("new content")

        # First generation
        generate_files(repo_dir, output_dir=output_dir)

        # Modify the generated file
        (output_dir / "test" / "file.txt").write_text("old content")

        # Second generation with skip_if_file_exists should not overwrite
        generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        assert (output_dir / "test" / "file.txt").read_text() == "old content"

    # Test with binary file
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Create a binary file
        binary_content = b'\x00\x01\x02\x03'
        binary_file = repo_dir / "{{cookiecutter.project_name}}" / "binary.bin"
        binary_file.write_bytes(binary_content)

        generate_files(repo_dir, output_dir=output_dir)
        output_binary = output_dir / "test" / "binary.bin"
        assert output_binary.exists()
        assert output_binary.read_bytes() == binary_content

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        context = {
            "project_name": "test",
            "_copy_without_render": ["*.bin", "static/*"]
        }
        (repo_dir / "cookiecutter.json").write_text(json.dumps(context))
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Create files that should be copied without rendering
        (repo_dir / "{{cookiecutter.project_name}}" / "file.bin").write_text("{{cookiecutter.project_name}}")
        (repo_dir / "{{cookiecutter.project_name}}" / "static").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "static" / "file.txt").write_text("{{cookiecutter.project_name}}")

        generate_files(repo_dir, output_dir=output_dir)
        assert (output_dir / "test" / "file.bin").read_text() == "{{cookiecutter.project_name}}"
        assert (output_dir / "test" / "static" / "file.txt").read_text() == "{{cookiecutter.project_name}}"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Create pre and post hooks
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "pre_gen_project.py").write_text("print('pre hook executed')")
        (hooks_dir / "post_gen_project.py").write_text("print('post hook executed')")

        # Test with hooks enabled
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)
        assert (output_dir / "test").exists()

        # Test with hooks disabled
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=False)
        assert (output_dir / "test").exists()

    # Test with undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context['cookiecutter']['project_name'] == 'My Project'
    assert context['cookiecutter']['project_slug'] == 'my_project'

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context('tests/test-data/cookiecutter.json', default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context('tests/test-data/cookiecutter.json', extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with both default and extra context
    context = generate_context('tests/test-data/cookiecutter.json', default_context, extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-data/nonexistent.json')


# LLM-generated content at query #51
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
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


# LLM-generated content at query #52
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup test environment
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a test template file
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ name }}!")

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

def test_generate_file_binary(tmp_path):
    # Setup test environment
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a binary test file
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    binary_file = template_dir / "test.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')

    # Create context
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test binary file copy
    generate_file(
        str(project_dir),
        "test.bin",
        context,
        env,
        skip_if_file_exists=False
    )

    # Verify the binary file was copied correctly
    output_file = project_dir / "test.bin"
    assert output_file.exists()
    assert output_file.read_bytes() == b'\x00\x01\x02\x03'

def test_generate_file_skip_if_exists(tmp_path):
    # Setup test environment
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a test template file
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ name }}!")

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

    # Create existing output file
    output_file = project_dir / "test.txt"
    output_file.write_text("Existing content")

    # Test file generation with skip_if_file_exists=True
    generate_file(
        str(project_dir),
        "test.txt",
        context,
        env,
        skip_if_file_exists=True
    )

    # Verify the existing file was not modified
    assert output_file.read_text() == "Existing content"

def test_generate_file_undefined_variable(tmp_path):
    # Setup test environment
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a test template file with undefined variable
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_file = template_dir / "test.txt"
    template_file.write_text("Hello, {{ undefined_var }}!")

    # Create context without the undefined variable
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    # Test that UndefinedError is raised
    with pytest.raises(UndefinedError):
        generate_file(
            str(project_dir),
            "test.txt",
            context,
            env,
            skip_if_file_exists=False
        )


# LLM-generated content at query #53
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'tests/test-fixtures/context.json'
    context = generate_context(context_file)
    assert context == {'cookiecutter': {'project_name': 'test_project'}}

    # Test with a non-existent context file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent.json')

    # Test with an invalid JSON context file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-fixtures/invalid_context.json')

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with both default_context and extra_context
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'


# LLM-generated content at query #54
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'tests/test_data/cookiecutter.json'
    context = generate_context(context_file)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'test_project'

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test_data/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test_data/nonexistent.json')


# LLM-generated content at query #55
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "tests/mocks/valid-template"
    output_dir = "tests/mocks/output"
    context = {"cookiecutter": {"project_name": "test_project"}}
    expected_project_dir = os.path.join(output_dir, "test_project")

    # Test
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    # Assert
    assert result == expected_project_dir
    assert os.path.exists(expected_project_dir)
    assert os.path.exists(os.path.join(expected_project_dir, "README.md"))
    assert os.path.exists(os.path.join(expected_project_dir, "setup.py"))

    # Cleanup
    shutil.rmtree(expected_project_dir)


# LLM-generated content at query #56
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        assert os.path.exists(project_dir)
        assert 'test_project' in project_dir

    # Test with context override
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        context = {'project_name': 'overridden_project'}
        project_dir = generate_files(repo_dir, context=context, output_dir=temp_dir)
        assert os.path.exists(project_dir)
        assert 'overridden_project' in project_dir

    # Test with existing output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        with pytest.raises(OutputDirExistsException):
            generate_files(repo_dir, output_dir=temp_dir)

    # Test with overwrite_if_exists=True
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        project_dir = generate_files(repo_dir, output_dir=temp_dir, overwrite_if_exists=True)
        assert os.path.exists(project_dir)

    # Test with template files
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        assert os.path.exists(os.path.join(project_dir, 'test.txt'))
        with open(os.path.join(project_dir, 'test.txt')) as f:
            assert 'Hello, test_project!' in f.read()

    # Test with copy_only paths
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project', '_copy_without_render': ['*.md']}, f)

        template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, 'README.md'), 'w') as f:
            f.write('This is a {{cookiecutter.project_name}} project.')

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        assert os.path.exists(os.path.join(project_dir, 'README.md'))
        with open(os.path.join(project_dir, 'README.md')) as f:
            assert 'This is a {{cookiecutter.project_name}} project.' in f.read()

    # Test with hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        hooks_dir = os.path.join(repo_dir, 'hooks')
        os.makedirs(hooks_dir)
        with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('import os\nos.makedirs("pre_hook_dir")')

        project_dir = generate_files(repo_dir, output_dir=temp_dir, accept_hooks=True)
        assert os.path.exists(os.path.join(project_dir, 'pre_hook_dir'))

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        project_dir = generate_files(repo_dir, output_dir=temp_dir)
        with open(os.path.join(project_dir, 'test.txt'), 'w') as f:
            f.write('Existing content')

        generate_files(repo_dir, output_dir=temp_dir, skip_if_file_exists=True)
        with open(os.path.join(project_dir, 'test.txt')) as f:
            assert 'Existing content' in f.read()

    # Test with keep_project_on_failure
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=temp_dir, keep_project_on_failure=True)
        assert os.path.exists(project_dir)


# LLM-generated content at query #57
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


# LLM-generated content at query #58
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'project_name': 'test_project', '_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello, {{ cookiecutter.project_name }}!')

    # Test
    generate_file(project_dir, infile, context, env)

    # Verify
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test_project!'

    # Cleanup
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #59
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
    assert context == {'cookiecutter': {'project_name': 'default_project'}}

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context == {'cookiecutter': {'project_name': 'extra_project'}}

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    os.remove(context_file)


# LLM-generated content at query #60
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path('tests/fake-repo-pre')
    output_dir = Path('tests/output')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
            'author': 'Test Author',
            'email': 'test@example.com',
            'version': '0.1.0',
            '_copy_without_render': ['*.bin', 'static/*'],
            '_new_lines': '\n'
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
        keep_project_on_failure=True
    )

    # Assert
    assert Path(result).exists()
    assert Path(result, 'README.md').exists()
    assert Path(result, 'setup.py').exists()
    assert Path(result, 'static').exists()
    assert Path(result, 'static', 'test.bin').exists()

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #61
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        context = generate_context(context_file)
        assert context == {'cookiecutter': {'project_name': 'test_project'}}

    # Test with default_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        default_context = {'project_name': 'default_project'}
        context = generate_context(context_file, default_context)
        assert context == {'cookiecutter': {'project_name': 'default_project'}}

    # Test with extra_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        extra_context = {'project_name': 'extra_project'}
        context = generate_context(context_file, extra_context=extra_context)
        assert context == {'cookiecutter': {'project_name': 'extra_project'}}

    # Test with invalid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w') as f:
            f.write('invalid json')

        with pytest.raises(ContextDecodingException):
            generate_context(context_file)

    # Test with non-existent JSON file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent.json')


# LLM-generated content at query #62
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.flush()
        context = generate_context(f.name)
        assert context == {'cookiecutter': {'key': 'value'}}
        os.unlink(f.name)

    # Test with a non-existent JSON file
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')

    # Test with an invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('invalid json')
        f.flush()
        with pytest.raises(ContextDecodingException):
            generate_context(f.name)
        os.unlink(f.name)

    # Test with default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.flush()
        context = generate_context(f.name, default_context={'key': 'default'})
        assert context == {'cookiecutter': {'key': 'default'}}
        os.unlink(f.name)

    # Test with extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.flush()
        context = generate_context(f.name, extra_context={'key': 'extra'})
        assert context == {'cookiecutter': {'key': 'extra'}}
        os.unlink(f.name)


# LLM-generated content at query #63
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "test_template"
    context = {"project_name": "test_project", "author": "test_author"}
    output_dir = "test_output"

    # Create a test template directory
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "{{cookiecutter.project_name}}"), exist_ok=True)

    # Create a test file in the template
    test_file_path = os.path.join(repo_dir, "{{cookiecutter.project_name}}", "test_file.txt")
    with open(test_file_path, "w") as f:
        f.write("Hello, {{cookiecutter.author}}!")

    # Create a cookiecutter.json file
    cookiecutter_json = {
        "project_name": "default_project",
        "author": "default_author"
    }
    with open(os.path.join(repo_dir, "cookiecutter.json"), "w") as f:
        json.dump(cookiecutter_json, f)

    # Execute
    result = generate_files(repo_dir, context, output_dir)

    # Verify
    expected_output_dir = os.path.join(output_dir, "test_project")
    assert os.path.exists(expected_output_dir)

    expected_file_path = os.path.join(expected_output_dir, "test_file.txt")
    assert os.path.exists(expected_file_path)

    with open(expected_file_path, "r") as f:
        content = f.read()
    assert content == "Hello, test_author!"

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #64
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    env = Environment(loader=FileSystemLoader('tests/templates'))

    # Create test template file
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.project_name }}!')

    # Test file generation
    generate_file(project_dir, infile, context, env)

    # Verify file was created and rendered correctly
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test_project!'

    # Clean up
    os.remove(infile)
    if os.path.exists(outfile):
        os.remove(outfile)


# LLM-generated content at query #65
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path('tests/test-templates/basic')
    output_dir = Path('tests/test-output')
    context = {'project_name': 'test_project', 'project_slug': 'test_slug'}

    # Test basic generation
    project_dir = generate_files(repo_dir, context, output_dir)
    assert project_dir.exists()
    assert (Path(project_dir) / 'test_slug').exists()

    # Test overwrite
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert project_dir.exists()

    # Test skip if exists
    with (Path(project_dir) / 'test_slug' / 'test_file.txt').open('w') as f:
        f.write('existing content')
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    with (Path(project_dir) / 'test_slug' / 'test_file.txt').open('r') as f:
        assert f.read() == 'existing content'

    # Test hooks
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert project_dir.exists()

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #66
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


# LLM-generated content at query #67
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


# LLM-generated content at query #68
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

    # Test basic generation
    output_dir = tmp_path / "output"
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "test_project" / "file.txt").exists()
    assert (output_dir / "test_project" / "file.txt").read_text() == "content"

    # Test with context override
    result = generate_files(
        repo_dir=str(repo_dir),
        context={"project_name": "override_project"},
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "override_project" / "file.txt").exists()

    # Test skip_if_file_exists
    (output_dir / "override_project" / "file.txt").write_text("new content")
    result = generate_files(
        repo_dir=str(repo_dir),
        context={"project_name": "override_project"},
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True
    )

    assert (output_dir / "override_project" / "file.txt").read_text() == "new content"

    # Test hooks
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("""
import os
with open('hook_test.txt', 'w') as f:
    f.write('pre hook')
""")
    (repo_dir / "hooks" / "post_gen_project.py").write_text("""
import os
with open('hook_test.txt', 'a') as f:
    f.write('post hook')
""")

    result = generate_files(
        repo_dir=str(repo_dir),
        context={"project_name": "hook_project"},
        output_dir=output_dir,
        overwrite_if_exists=True,
        accept_hooks=True
    )

    assert (output_dir / "hook_project" / "hook_test.txt").read_text() == "pre hookpost hook"

    # Test error cases
    (repo_dir / "bad_template" / "{{cookiecutter.undefined_var}}").mkdir()
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context={"project_name": "test"},
            output_dir=output_dir,
            overwrite_if_exists=True
        )


# LLM-generated content at query #69
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'tests/test_data/cookiecutter.json'
    context = generate_context(context_file)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'test_project'
    assert context['cookiecutter']['project_slug'] == 'test_project'

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test_data/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test_data/nonexistent.json')


# LLM-generated content at query #70
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    import tempfile
    import os
    from jinja2 import Environment

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = os.path.join(temp_dir, 'project')
        os.makedirs(project_dir)

        # Create a test input file
        infile = os.path.join(temp_dir, 'test.txt')
        with open(infile, 'w', encoding='utf-8') as f:
            f.write('Hello {{ name }}!')

        # Create context
        context = {
            'cookiecutter': {
                '_copy_without_render': [],
                '_new_lines': '\n'
            },
            'name': 'World'
        }

        # Create Jinja2 environment
        env = Environment()

        # Test file generation
        generate_file(
            project_dir=project_dir,
            infile=infile,
            context=context,
            env=env,
            skip_if_file_exists=False
        )

        # Verify output file exists and has correct content
        outfile = os.path.join(project_dir, 'test.txt')
        assert os.path.exists(outfile)
        with open(outfile, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == 'Hello World!'

        # Test binary file handling
        binary_file = os.path.join(temp_dir, 'binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'test binary content')

        generate_file(
            project_dir=project_dir,
            infile=binary_file,
            context=context,
            env=env,
            skip_if_file_exists=False
        )

        binary_outfile = os.path.join(project_dir, 'binary.bin')
        assert os.path.exists(binary_outfile)
        with open(binary_outfile, 'rb') as f:
            binary_content = f.read()
        assert binary_content == b'test binary content'

        # Test skip_if_file_exists
        generate_file(
            project_dir=project_dir,
            infile=infile,
            context=context,
            env=env,
            skip_if_file_exists=True
        )

        # File should still have original content
        with open(outfile, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == 'Hello World!'


# LLM-generated content at query #71
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
    shutil.rmtree(project_dir)


# LLM-generated content at query #72
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path("tests/test-templates/basic-template")
    output_dir = Path("tests/test-output")
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
            "author": "Test Author",
        }
    }

    # Test
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
    )

    # Assert
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project").exists()
    assert (output_dir / "test_project" / "README.md").exists()
    assert (output_dir / "test_project" / "setup.py").exists()

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #73
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        assert result_dir == Path(tmpdir) / 'test_project'
        assert created is True
        assert result_dir.exists()

    # Test directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        # Create the directory first
        test_dir = Path(tmpdir) / 'test_project'
        test_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=False)

    # Test directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dirname = '{{ cookiecutter.project_name }}'
        output_dir = Path(tmpdir)

        # Create the directory first
        test_dir = Path(tmpdir) / 'test_project'
        test_dir.mkdir()
        test_file = test_dir / 'test.txt'
        test_file.write_text('original content')

        result_dir, created = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        assert result_dir == Path(tmpdir) / 'test_project'
        assert created is False
        assert result_dir.exists()
        assert not test_file.exists()  # Directory should be empty after overwrite

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


# LLM-generated content at query #74
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    env = Environment(loader=FileSystemLoader('.'))
    env.globals.update({'cookiecutter': context['cookiecutter']})

    # Create test template file
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.project_name }}!')

    # Test file generation
    generate_file(project_dir, infile, context, env)

    # Check if file was created and rendered correctly
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test_project!'

    # Cleanup
    os.remove(infile)
    if os.path.exists(outfile):
        os.remove(outfile)


# LLM-generated content at query #75
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    repo_dir = "tests/test-templates/basic"
    output_dir = "tests/output/basic"
    context = {"project_name": "test_project"}

    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert "test_project" in result

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True
    )

    # Test with hooks
    repo_dir_with_hooks = "tests/test-templates/with-hooks"
    output_dir_hooks = "tests/output/with-hooks"

    result = generate_files(
        repo_dir=repo_dir_with_hooks,
        context=context,
        output_dir=output_dir_hooks,
        overwrite_if_exists=True,
        accept_hooks=True
    )

    assert os.path.exists(result)

    # Test with undefined variable in template
    repo_dir_bad = "tests/test-templates/undefined-variable"
    output_dir_bad = "tests/output/undefined-variable"

    try:
        generate_files(
            repo_dir=repo_dir_bad,
            context=context,
            output_dir=output_dir_bad,
            overwrite_if_exists=True
        )
        assert False, "Should have raised UndefinedVariableInTemplate"
    except UndefinedVariableInTemplate:
        pass

    # Test with output directory already exists
    output_dir_exists = "tests/output/exists"
    os.makedirs(output_dir_exists, exist_ok=True)

    try:
        generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir_exists,
            overwrite_if_exists=False
        )
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass

    # Clean up
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(output_dir_hooks, ignore_errors=True)
    shutil.rmtree(output_dir_bad, ignore_errors=True)
    shutil.rmtree(output_dir_exists, ignore_errors=True)


# LLM-generated content at query #76
#--------------------------

```python
def test_generate_files():
    # Setup test data
    repo_dir = Path("tests/test-templates/simple")
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = Path("tests/output")
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = False

    # Call the function
    result = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists,
        skip_if_file_exists,
        accept_hooks,
        keep_project_on_failure,
    )

    # Assertions
    assert isinstance(result, str)
    assert Path(result).exists()
    assert Path(result).is_dir()
    assert Path(result).name == "test_project"
    assert Path(result, "README.md").exists()
    assert Path(result, "setup.py").exists()

    # Cleanup
    shutil.rmtree(result)


# LLM-generated content at query #77
#--------------------------

```python
def test_generate_context():
    # Test with valid context file
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    context = generate_context(context_file)
    assert context == {'test_context': {'key': 'value'}}
    os.remove(context_file)

    # Test with default context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context=default_context)
    assert context['test_context']['key'] == 'default_value'

    # Test with extra context
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
        generate_context('non_existent_file.json')


# LLM-generated content at query #78
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    context = generate_context(context_file)
    assert context == {'cookiecutter': {'key': 'value'}}

    # Test with default_context
    default_context = {'key': 'default_value'}
    context = generate_context(context_file, default_context)
    assert context == {'cookiecutter': {'key': 'default_value'}}

    # Test with extra_context
    extra_context = {'key': 'extra_value'}
    context = generate_context(context_file, None, extra_context)
    assert context == {'cookiecutter': {'key': 'extra_value'}}

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    # Clean up
    os.remove(context_file)


# LLM-generated content at query #79
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory overwriting
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert created

        # Try to create again with overwrite
        dir_to_create2, created2 = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create2 == dir_to_create
        assert not created2

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test existing directory without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'tests/data/cookiecutter.json'
    context = generate_context(context_file)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'test_project'

    # Test with a non-existent context file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent.json')

    # Test with an invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/data/invalid.json')

    # Test with default context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra context
    extra_context = {'project_name': 'extra_project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'

    # Test with both default and extra context
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['project_name'] == 'extra_project'


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory rendering and creation
    dirname = "my_{{ cookiecutter.project_name }}_project"
    context = {'cookiecutter': {'project_name': 'test'}}
    output_dir = Path('/tmp/test_output')
    environment = Environment()

    result_dir, created = render_and_create_dir(dirname, context, output_dir, environment)

    assert result_dir == output_dir / "my_test_project"
    assert created is True
    assert result_dir.exists()

    # Clean up
    rmtree(result_dir)

    # Test with existing directory (should raise exception)
    existing_dir = output_dir / "existing"
    existing_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("existing", context, output_dir, environment)

    # Clean up
    rmtree(existing_dir)

    # Test with overwrite_if_exists=True
    existing_dir.mkdir(parents=True, exist_ok=True)
    result_dir, created = render_and_create_dir("existing", context, output_dir, environment, overwrite_if_exists=True)

    assert result_dir == existing_dir
    assert created is False
    assert result_dir.exists()

    # Clean up
    rmtree(result_dir)

    # Test with empty directory name (should raise exception)
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Test with template rendering error
    with pytest.raises(UndefinedVariableInTemplate):
        render_and_create_dir("my_{{ cookiecutter.nonexistent }}_project", context, output_dir, environment)


# LLM-generated content at query #3
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

    # Test overwrite with list (choice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': 'choice2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice1', 'choice3']

    # Test overwrite with list (invalid choice)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': 'invalid_choice'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test overwrite with list (multichoice variable)
    context = {'var1': ['choice1', 'choice2', 'choice3']}
    overwrite_context = {'var1': ['choice2', 'choice3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == ['choice2', 'choice3']

    # Test overwrite with list (invalid multichoice)
    context = {'var1': ['choice1', 'choice2']}
    overwrite_context = {'var1': ['choice1', 'invalid_choice']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_choice" in str(e)

    # Test overwrite with dict
    context = {'var1': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'var1': {'key1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'value2'

    # Test overwrite with dict (new key)
    context = {'var1': {'key1': 'value1'}}
    overwrite_context = {'var1': {'key1': 'new_value1', 'key2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['key1'] == 'new_value1'
    assert context['var1']['key2'] == 'new_value2'

    # Test overwrite with boolean (valid)
    context = {'var1': True}
    overwrite_context = {'var1': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] is True

    # Test overwrite with boolean (invalid)
    context = {'var1': True}
    overwrite_context = {'var1': 'invalid_bool'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid_bool" in str(e)

    # Test overwrite with nested dict
    context = {'var1': {'nested': {'key1': 'value1'}}}
    overwrite_context = {'var1': {'nested': {'key1': 'new_value1'}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1']['nested']['key1'] == 'new_value1'


# LLM-generated content at query #4
#--------------------------

```python
def test_is_copy_only_path():
    # Test case 1: Path matches a pattern in context
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('docs/index.md', context) is True

    # Test case 2: Path does not match any pattern in context
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

    # Test case 3: Context does not have '_copy_without_render' key
    context_no_key = {
        'cookiecutter': {}
    }
    assert is_copy_only_path('file.txt', context_no_key) is False

    # Test case 4: Context does not have 'cookiecutter' key
    context_no_cookiecutter = {}
    assert is_copy_only_path('file.txt', context_no_cookiecutter) is False

    # Test case 5: Empty path
    assert is_copy_only_path('', context) is False


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_context():
    # Test with default parameters
    context = generate_context()
    assert isinstance(context, OrderedDict)
    assert 'cookiecutter' in context

    # Test with a valid context file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.flush()
        context = generate_context(f.name)
        assert context['cookiecutter']['key'] == 'value'
        os.unlink(f.name)

    # Test with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('invalid json')
        f.flush()
        with pytest.raises(ContextDecodingException):
            generate_context(f.name)
        os.unlink(f.name)

    # Test with default_context and extra_context
    default_context = {'key1': 'default_value'}
    extra_context = {'key1': 'extra_value', 'key2': 'extra_value2'}
    context = generate_context(extra_context=extra_context, default_context=default_context)
    assert context['cookiecutter']['key1'] == 'extra_value'
    assert context['cookiecutter']['key2'] == 'extra_value2'

    # Test with multichoice variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'choices': ['a', 'b', 'c']}, context_file)
    context_file.flush()
    extra_context = {'choices': ['a', 'c']}
    context = generate_context(context_file.name, extra_context=extra_context)
    assert context['cookiecutter']['choices'] == ['a', 'c']
    context_file.close()
    os.unlink(context_file.name)

    # Test with invalid multichoice variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'choices': ['a', 'b', 'c']}, context_file)
    context_file.flush()
    extra_context = {'choices': ['a', 'd']}
    with pytest.raises(ValueError):
        generate_context(context_file.name, extra_context=extra_context)
    context_file.close()
    os.unlink(context_file.name)

    # Test with choice variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'choice': ['a', 'b', 'c']}, context_file)
    context_file.flush()
    extra_context = {'choice': 'b'}
    context = generate_context(context_file.name, extra_context=extra_context)
    assert context['cookiecutter']['choice'] == ['b', 'a', 'c']
    context_file.close()
    os.unlink(context_file.name)

    # Test with invalid choice variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'choice': ['a', 'b', 'c']}, context_file)
    context_file.flush()
    extra_context = {'choice': 'd'}
    with pytest.raises(ValueError):
        generate_context(context_file.name, extra_context=extra_context)
    context_file.close()
    os.unlink(context_file.name)

    # Test with boolean variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'bool_var': True}, context_file)
    context_file.flush()
    extra_context = {'bool_var': 'yes'}
    context = generate_context(context_file.name, extra_context=extra_context)
    assert context['cookiecutter']['bool_var'] is True
    context_file.close()
    os.unlink(context_file.name)

    # Test with invalid boolean variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'bool_var': True}, context_file)
    context_file.flush()
    extra_context = {'bool_var': 'invalid'}
    with pytest.raises(ValueError):
        generate_context(context_file.name, extra_context=extra_context)
    context_file.close()
    os.unlink(context_file.name)

    # Test with dictionary variable
    context_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'dict_var': {'key1': 'value1', 'key2': 'value2'}}, context_file)
    context_file.flush()
    extra_context = {'dict_var': {'key1': 'new_value1'}}
    context = generate_context(context_file.name, extra_context=extra_context)
    assert context['cookiecutter']['dict_var']['key1'] == 'new_value1'
    assert context['cookiecutter']['dict_var']['key2'] == 'value2'
    context_file.close()
    os.unlink(context_file.name)


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    dirname = "test_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    result_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result_path == output_dir / "test_dir"
    assert created is True
    assert result_path.exists()

    # Test with template rendering
    dirname = "{{ test }}_dir"
    result_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result_path == output_dir / "value_dir"
    assert created is True
    assert result_path.exists()

    # Test overwrite_if_exists=True
    result_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result_path == output_dir / "value_dir"
    assert created is False  # Directory already exists
    assert result_path.exists()

    # Test overwrite_if_exists=False with existing directory
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=False)

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Cleanup
    shutil.rmtree(output_dir / "test_dir")
    shutil.rmtree(output_dir / "value_dir")


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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

    # Test file generation
    generate_file(project_dir, infile, context, env)

    # Verify the output file was created and rendered correctly
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Cleanup
    os.remove(infile)
    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test with simple overwrite
    context = {'var1': 'value1', 'var2': 'value2'}
    overwrite_context = {'var1': 'new_value1'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert context['var2'] == 'value2'

    # Test with new variable (should be ignored)
    context = {'var1': 'value1'}
    overwrite_context = {'var1': 'new_value1', 'var2': 'new_value2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['var1'] == 'new_value1'
    assert 'var2' not in context

    # Test with list choice variable
    context = {'choice_var': ['option1', 'option2', 'option3']}
    overwrite_context = {'choice_var': 'option2'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['choice_var'] == ['option2', 'option1', 'option3']

    # Test with invalid choice variable
    context = {'choice_var': ['option1', 'option2']}
    overwrite_context = {'choice_var': 'invalid_option'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test with multichoice variable
    context = {'multi_var': ['option1', 'option2', 'option3']}
    overwrite_context = {'multi_var': ['option1', 'option3']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['multi_var'] == ['option1', 'option3']

    # Test with invalid multichoice variable
    context = {'multi_var': ['option1', 'option2']}
    overwrite_context = {'multi_var': ['option1', 'invalid_option']}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test with boolean variable
    context = {'bool_var': True}
    overwrite_context = {'bool_var': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['bool_var'] is True

    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'no'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['bool_var'] is False

    # Test with invalid boolean variable
    context = {'bool_var': True}
    overwrite_context = {'bool_var': 'invalid_bool'}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)

    # Test with nested dictionary
    context = {'nested': {'var1': 'value1', 'var2': 'value2'}}
    overwrite_context = {'nested': {'var1': 'new_value1'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['nested']['var1'] == 'new_value1'
    assert context['nested']['var2'] == 'value2'

    # Test with new nested variable
    context = {'nested': {'var1': 'value1'}}
    overwrite_context = {'nested': {'var1': 'new_value1', 'var2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context['nested']['var1'] == 'new_value1'
    assert context['nested']['var2'] == 'new_value2'


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context():
    # Test basic context generation from a JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'Test Author'}, f)

        context = generate_context(context_file)
        assert context == {'cookiecutter': {'project_name': 'test_project', 'author': 'Test Author'}}

    # Test context generation with default_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'Test Author'}, f)

        default_context = {'project_name': 'default_project'}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == 'default_project'
        assert context['cookiecutter']['author'] == 'Test Author'

    # Test context generation with extra_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'Test Author'}, f)

        extra_context = {'project_name': 'extra_project'}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == 'extra_project'
        assert context['cookiecutter']['author'] == 'Test Author'

    # Test context generation with invalid JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('{invalid json}')

        with pytest.raises(ContextDecodingException):
            generate_context(context_file)

    # Test context generation with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')


# LLM-generated content at query #11
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("Hello")

        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Verify output
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "Hello"

    # Test with context override
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "default"}')
        (repo_dir / "{{cookiecutter.name}}.txt").write_text("Content")

        # Generate with override
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            extra_context={"name": "custom"},
            overwrite_if_exists=True
        )

        assert (output_dir / "custom.txt").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "{{cookiecutter.name}}.txt").write_text("Original")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Modify template
        (repo_dir / "{{cookiecutter.name}}.txt").write_text("Modified")

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        # Should keep original content
        assert (output_dir / "test.txt").read_text() == "Original"

    # Test hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with hooks
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "hooks").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text(
            "import os\nwith open('pre_hook.txt', 'w') as f: f.write('pre')"
        )
        (repo_dir / "hooks" / "post_gen_project.py").write_text(
            "import os\nwith open('post_hook.txt', 'w') as f: f.write('post')"
        )

        # Generate with hooks
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True,
            overwrite_if_exists=True
        )

        assert (Path(result) / "pre_hook.txt").exists()
        assert (Path(result) / "post_hook.txt").exists()

    # Test error handling for undefined variables
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with undefined variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "{{cookiecutter.undefined}}.txt").write_text("Content")

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir,
                overwrite_if_exists=True
            )

    # Test copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with copy_without_render
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text(
            '{"name": "test", "_copy_without_render": ["*.bin"]}'
        )
        (repo_dir / "{{cookiecutter.name}}.bin").write_bytes(b"\x00\x01\x02")

        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Should copy binary file without rendering
        assert (output_dir / "test.bin").exists()
        assert (output_dir / "test.bin").read_bytes() == b"\x00\x01\x02"


# LLM-generated content at query #12
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory creation with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert created

        # Create again with overwrite
        dir_to_create2, created2 = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create2.exists()
        assert not created2  # Should not be created again

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test existing directory without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created

        # Try to create again without overwrite
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = '/tmp/test_project'
    infile = 'test_template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    template_dir = '/tmp/test_template'
    os.makedirs(template_dir, exist_ok=True)

    # Create a test template file
    with open(os.path.join(template_dir, infile), 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')

    # Test file generation
    with work_in(template_dir):
        generate_file(project_dir, infile, context, env)

    # Verify the output file was created and contains rendered content
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test!'

    # Test binary file handling
    binary_file = 'test_binary.bin'
    with open(os.path.join(template_dir, binary_file), 'wb') as f:
        f.write(b'\x00\x01\x02\x03')

    generate_file(project_dir, binary_file, context, env)
    binary_outfile = os.path.join(project_dir, binary_file)
    assert os.path.exists(binary_outfile)
    with open(binary_outfile, 'rb') as f:
        binary_content = f.read()
    assert binary_content == b'\x00\x01\x02\x03'

    # Test skip_if_file_exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, 'r') as f:
        content = f.read()
    assert content == 'Hello, test!'  # Content should not change

    # Cleanup
    shutil.rmtree(project_dir, ignore_errors=True)
    shutil.rmtree(template_dir, ignore_errors=True)


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mock_project_template/cookiecutter.json')
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'My Project'

    # Test with default context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/mock_project_template/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(
        'tests/mock_project_template/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mock_project_template/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mock_project_template/nonexistent.json')


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_file():
    # Setup
    import tempfile
    import os
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = os.path.join(tmpdir, 'project')
        os.makedirs(project_dir)

        # Create a test template file
        template_content = "Hello, {{ name }}!"
        template_file = os.path.join(tmpdir, 'template.txt')
        with open(template_file, 'w') as f:
            f.write(template_content)

        # Create context
        context = {'name': 'World'}

        # Create Jinja2 environment
        env = Environment()

        # Test file generation
        generate_file(
            project_dir=project_dir,
            infile='template.txt',
            context={'cookiecutter': context},
            env=env
        )

        # Verify the file was created and rendered correctly
        output_file = os.path.join(project_dir, 'template.txt')
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            content = f.read()
        assert content == "Hello, World!"

        # Test binary file handling
        binary_file = os.path.join(tmpdir, 'binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'test binary content')

        generate_file(
            project_dir=project_dir,
            infile='binary.bin',
            context={'cookiecutter': context},
            env=env
        )

        output_binary = os.path.join(project_dir, 'binary.bin')
        assert os.path.exists(output_binary)
        with open(output_binary, 'rb') as f:
            binary_content = f.read()
        assert binary_content == b'test binary content'

        # Test skip_if_file_exists
        generate_file(
            project_dir=project_dir,
            infile='template.txt',
            context={'cookiecutter': context},
            env=env,
            skip_if_file_exists=True
        )

        # File should not be overwritten
        with open(output_file, 'r') as f:
            content = f.read()
        assert content == "Hello, World!"

        # Test newline handling
        template_with_newlines = "Line 1\nLine 2\r\nLine 3"
        template_newlines_file = os.path.join(tmpdir, 'newlines.txt')
        with open(template_newlines_file, 'w') as f:
            f.write(template_with_newlines)

        context['_new_lines'] = '\n'
        generate_file(
            project_dir=project_dir,
            infile='newlines.txt',
            context={'cookiecutter': context},
            env=env
        )

        output_newlines = os.path.join(project_dir, 'newlines.txt')
        with open(output_newlines, 'r') as f:
            newlines_content = f.read()
        assert newlines_content == "Line 1\nLine 2\nLine 3"


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir, exist_ok=True)

    # Create a test input file
    infile = "test_file.txt"
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello, {{ name }}!")

    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "name": "World"
    }

    env = Environment(loader=FileSystemLoader("."))

    # Test normal file generation
    generate_file(project_dir, infile, context, env)

    # Verify the file was created and rendered correctly
    expected_file = os.path.join(project_dir, infile)
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello, World!"

    # Test binary file handling
    binary_infile = "test_binary.bin"
    with open(binary_infile, "wb") as f:
        f.write(b"binary content")

    generate_file(project_dir, binary_infile, context, env)

    expected_binary_file = os.path.join(project_dir, binary_infile)
    assert os.path.exists(expected_binary_file)
    with open(expected_binary_file, "rb") as f:
        binary_content = f.read()
    assert binary_content == b"binary content"

    # Test skip_if_file_exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

    # Verify the file wasn't overwritten
    with open(expected_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello, World!"

    # Test file with template syntax error
    bad_infile = "bad_template.txt"
    with open(bad_infile, "w", encoding="utf-8") as f:
        f.write("{% if %}")

    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, bad_infile, context, env)

    # Test undefined variable in template
    bad_context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        }
    }

    with pytest.raises(UndefinedError):
        generate_file(project_dir, infile, bad_context, env)


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    output_dir = tmp_path / "output"

    # Test basic generation
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
        extra_context={"project_name": "another_project"},
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    assert (output_dir / "another_project" / "test.txt").exists()
    assert (output_dir / "another_project" / "test.txt").read_text() == "Hello, another_project!"

    # Test skip_if_file_exists
    (output_dir / "test_project" / "test.txt").write_text("Existing content")
    generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True
    )
    assert (output_dir / "test_project" / "test.txt").read_text() == "Existing content"

    # Test hooks
    (repo_dir / "hooks" / "pre_gen_project.py").write_text("""
import os
with open(os.path.join(project_dir, 'hook_test.txt'), 'w') as f:
    f.write('pre hook executed')
""")
    result = generate_files(
        repo_dir=str(repo_dir),
        output_dir=output_dir,
        overwrite_if_exists=True,
        accept_hooks=True
    )
    assert (output_dir / "test_project" / "hook_test.txt").exists()
    assert (output_dir / "test_project" / "hook_test.txt").read_text() == "pre hook executed"

    # Test error handling
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "{{undefined_var}}"}')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )


# LLM-generated content at query #19
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
    shutil.rmtree(project_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory creation with existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        # Should raise exception when directory exists
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                context,
                tmpdir,
                env,
                overwrite_if_exists=False
            )

        # Should overwrite when flag is set
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created  # Because it already existed

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                context,
                tmpdir,
                env
            )

    # Test undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test'}
        env = Environment()
        with pytest.raises(UndefinedVariableInTemplate):
            render_and_create_dir(
                '{{cookiecutter.nonexistent}}',
                context,
                tmpdir,
                env
            )


# LLM-generated content at query #21
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

    # Test with keep_project_on_failure
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'file.txt'), 'w') as f:
            f.write('test content {{ undefined_var }}')

        output_dir = os.path.join(tmpdir, 'output')
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir, keep_project_on_failure=True)
        assert os.path.exists(os.path.join(output_dir, 'test'))


# LLM-generated content at query #22
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"project_name": "test_project"}
    output_dir = Path(tempfile.mkdtemp())
    environment = Environment()

    # Test successful directory creation
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    assert result_dir.exists()
    assert created
    assert result_dir.name == "test_dir"

    # Test directory already exists without overwrite
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )

    # Test directory already exists with overwrite
    result_dir2, created2 = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert result_dir2.exists()
    assert not created2

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"project_name": "test_project"}
    output_dir = Path("/tmp")
    environment = Environment()

    # Test successful directory creation
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    assert result_path.exists()
    assert created is True
    assert result_path.name == "test_dir"

    # Test directory already exists without overwrite
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )

    # Test directory already exists with overwrite
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert result_path.exists()
    assert created is False

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Cleanup
    if result_path.exists():
        shutil.rmtree(result_path)


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_files():
    # Setup test data
    repo_dir = Path('tests/test-templates/simple')
    output_dir = Path('tests/test-output')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
            '_copy_without_render': ['*.txt']
        }
    }

    # Call the function
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=True
    )

    # Verify the result
    assert isinstance(result, str)
    assert Path(result).exists()
    assert Path(result).name == 'test_project'

    # Verify files were created
    assert Path(result, 'README.md').exists()
    assert Path(result, 'test.txt').exists()

    # Verify content
    with open(Path(result, 'README.md'), 'r') as f:
        content = f.read()
        assert 'test_project' in content

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with context override
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        result = generate_files(
            repo_dir=str(repo_dir),
            context={"cookiecutter": {"project_name": "override"}},
            output_dir=output_dir
        )

        assert (output_dir / "override").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("old")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Modify template
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("new")

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        assert (output_dir / "test" / "file.txt").read_text() == "old"

    # Test hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("""
import os
with open(os.path.join('{{cookiecutter.project_name}}', 'hook.txt'), 'w') as f:
    f.write('pre')
""")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True
        )

        assert (output_dir / "test" / "hook.txt").exists()
        assert (output_dir / "test" / "hook.txt").read_text() == "pre"

    # Test binary file handling
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Create a binary file
        binary_content = b'\x00\x01\x02\x03'
        (repo_dir / "{{cookiecutter.project_name}}" / "binary.bin").write_bytes(binary_content)

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir
        )

        assert (output_dir / "test" / "binary.bin").read_bytes() == binary_content

    # Test error handling for undefined variables
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir
            )


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'test_cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'project_name': 'test_project', 'author': 'test_author'}, f)

    context = generate_context(context_file)
    assert context == {'cookiecutter': {'project_name': 'test_project', 'author': 'test_author'}}

    # Test with default_context
    default_context = {'project_name': 'default_project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra_context
    extra_context = {'author': 'extra_author'}
    context = generate_context(context_file, None, extra_context)
    assert context['cookiecutter']['author'] == 'extra_author'

    # Test with invalid JSON
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')

    with pytest.raises(ContextDecodingException):
        generate_context(context_file)

    # Clean up
    os.remove(context_file)


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test template directory
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (template_dir / "{{cookiecutter.project_name}}").mkdir()
    (template_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello, {{cookiecutter.project_name}}!")

    # Test basic generation
    project_dir = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True
    )

    assert (Path(project_dir) / "test.txt").exists()
    assert (Path(project_dir) / "test.txt").read_text() == "Hello, test_project!"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        skip_if_file_exists=True
    )

    # Test with hooks
    (template_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
    (template_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

    project_dir = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True,
        accept_hooks=True
    )

    # Test with binary file
    binary_file = template_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    (template_dir / "cookiecutter.json").write_text('{"project_name": "test_project", "_copy_without_render": ["*.bin"]}')

    project_dir = generate_files(
        repo_dir=str(template_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=str(tmp_path),
        overwrite_if_exists=True
    )

    assert (Path(project_dir) / "binary.bin").exists()
    assert (Path(project_dir) / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'

    # Test with undefined variable
    (template_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (template_dir / "{{cookiecutter.undefined_var}}").mkdir()

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(template_dir),
            context={"cookiecutter": {"project_name": "test_project"}},
            output_dir=str(tmp_path),
            overwrite_if_exists=True
        )


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_file():
    # Setup
    project_dir = '/tmp/test_project'
    infile = 'test.txt'
    context = {'cookiecutter': {'_copy_without_render': [], '_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)

    # Create a test file
    test_file_content = "Hello, {{ name }}!"
    with open(infile, 'w') as f:
        f.write(test_file_content)

    # Test binary file
    binary_file = 'test.bin'
    with open(binary_file, 'wb') as f:
        f.write(b'test binary content')

    # Test render and create file
    context['name'] = 'World'
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == "Hello, World!"

    # Test binary file copy
    generate_file(project_dir, binary_file, context, env)
    with open(os.path.join(project_dir, binary_file), 'rb') as f:
        assert f.read() == b'test binary content'

    # Test skip if file exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == "Hello, World!"

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        generate_file(project_dir, '', context, env)

    # Test undefined variable in template
    context_with_undefined = {'cookiecutter': {'_copy_without_render': [], '_new_lines': None}}
    with pytest.raises(UndefinedVariableInTemplate):
        generate_file(project_dir, infile, context_with_undefined, env)

    # Cleanup
    os.remove(infile)
    os.remove(binary_file)
    shutil.rmtree(project_dir)


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mocks/cookiecutter.json')
    assert context == {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }

    # Test with default context
    default_context = {'project_name': 'default_project'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'default_project'

    # Test with extra context
    extra_context = {'author': 'extra_author'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['author'] == 'extra_author'

    # Test with both default and extra context
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        default_context=default_context,
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_name'] == 'default_project'
    assert context['cookiecutter']['author'] == 'extra_author'

    # Test with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mocks/invalid.json')

    # Test with missing file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mocks/nonexistent.json')


# LLM-generated content at query #30
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    dirname = "test_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp")
    environment = Environment()

    result_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result_path.exists()
    assert created is True
    assert result_path.name == "test_dir"

    # Test directory creation with template rendering
    dirname = "{{ test }}"
    context = {"test": "rendered_dir"}
    result_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result_path.exists()
    assert created is True
    assert result_path.name == "rendered_dir"

    # Test overwrite_if_exists=True
    dirname = "test_dir"
    context = {"test": "value"}
    result_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result_path.exists()
    assert created is False  # Directory already exists

    # Test overwrite_if_exists=False raises exception
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=False)

    # Test empty directory name raises exception
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Clean up
    if result_path.exists():
        shutil.rmtree(result_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Verify output
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with extra context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "default"}')
        (repo_dir / "{{cookiecutter.name}}").mkdir()
        (repo_dir / "{{cookiecutter.name}}" / "file.txt").write_text("{{cookiecutter.name}}")

        # Generate with extra context
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            extra_context={"name": "custom"},
            overwrite_if_exists=True
        )

        # Verify custom name was used
        assert (output_dir / "custom" / "file.txt").exists()
        assert (output_dir / "custom" / "file.txt").read_text() == "custom"

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "{{cookiecutter.name}}").mkdir()
        (repo_dir / "{{cookiecutter.name}}" / "file.txt").write_text("original")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Modify template
        (repo_dir / "{{cookiecutter.name}}" / "file.txt").write_text("modified")

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        # Verify original content was kept
        assert (output_dir / "test" / "file.txt").read_text() == "original"

    # Test hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with hooks
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "{{cookiecutter.name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("open('hook_marker', 'w').write('pre')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("open('hook_marker', 'a').write('post')")

        # Generate with hooks
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True,
            overwrite_if_exists=True
        )

        # Verify hooks ran
        assert (Path(result) / "hook_marker").exists()
        assert (Path(result) / "hook_marker").read_text() == "prepost"

    # Test error handling for undefined variables
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create template with undefined variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        # Should raise exception
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir,
                overwrite_if_exists=True
            )


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path('tests/test-template')
    context = {'project_name': 'test_project', 'author': 'test_author'}
    output_dir = Path('tests/output')

    # Ensure output directory is clean
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Test
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )

    # Assertions
    assert result == str(output_dir / 'test_project')
    assert (output_dir / 'test_project').exists()
    assert (output_dir / 'test_project' / 'README.md').exists()
    assert (output_dir / 'test_project' / 'src' / 'test_project').exists()

    # Cleanup
    shutil.rmtree(output_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json')
        assert context == {'test': {'key': 'value'}}

    # Test with a non-existent JSON file
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            generate_context('nonexistent.json')

    # Test with invalid JSON
    with patch('builtins.open', mock_open(read_data='{invalid json}')):
        with pytest.raises(ContextDecodingException):
            generate_context('invalid.json')

    # Test with default_context
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json', default_context={'key': 'new_value'})
        assert context == {'test': {'key': 'new_value'}}

    # Test with extra_context
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json', extra_context={'key': 'extra_value'})
        assert context == {'test': {'key': 'extra_value'}}

    # Test with both default_context and extra_context
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        context = generate_context('test.json', default_context={'key': 'default_value'}, extra_context={'key': 'extra_value'})
        assert context == {'test': {'key': 'extra_value'}}


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
def test_generate_files():
    # Setup
    repo_dir = "test_template"
    context = {"project_name": "test_project", "author": "test_author"}
    output_dir = "test_output"

    # Create a test template directory structure
    os.makedirs(repo_dir, exist_ok=True)
    with open(os.path.join(repo_dir, "cookiecutter.json"), "w") as f:
        json.dump({"project_name": "test_project", "author": "test_author"}, f)

    # Create a test file
    with open(os.path.join(repo_dir, "test_file.txt"), "w") as f:
        f.write("Hello, {{ cookiecutter.project_name }}!")

    # Test
    result = generate_files(repo_dir, context, output_dir)

    # Assertions
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert os.path.exists(os.path.join(result, "test_file.txt"))
    with open(os.path.join(result, "test_file.txt"), "r") as f:
        assert f.read() == "Hello, test_project!"

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #36
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'test_project'

    # Test with existing directory and overwrite_if_exists=False
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )

    # Test with existing directory and overwrite_if_exists=True
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()
        test_file = existing_dir / 'test.txt'
        test_file.write_text('old content')

        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert result_dir.exists()
        assert not created
        assert result_dir.name == 'test_project'
        assert not test_file.exists()  # Directory was overwritten

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test with template rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'my_project-1.0'


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_context():
    # Test with a valid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'test_author'}, f)

        context = generate_context(context_file)
        assert context == {'cookiecutter': {'project_name': 'test_project', 'author': 'test_author'}}

    # Test with default_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'test_author'}, f)

        default_context = {'project_name': 'default_project'}
        context = generate_context(context_file, default_context=default_context)
        assert context == {'cookiecutter': {'project_name': 'default_project', 'author': 'test_author'}}

    # Test with extra_context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump({'project_name': 'test_project', 'author': 'test_author'}, f)

        extra_context = {'project_name': 'extra_project'}
        context = generate_context(context_file, extra_context=extra_context)
        assert context == {'cookiecutter': {'project_name': 'extra_project', 'author': 'test_author'}}

    # Test with invalid JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        context_file = os.path.join(tmpdir, 'cookiecutter.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write('invalid json')

        with pytest.raises(ContextDecodingException):
            generate_context(context_file)

    # Test with non-existent JSON file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = "/tmp/test_project"
    infile = "test_template.txt"
    context = {
        "cookiecutter": {
            "_copy_without_render": [],
            "_new_lines": "\n"
        },
        "project_name": "test_project"
    }
    env = Environment(loader=FileSystemLoader("."))

    # Create test template file
    with open(infile, "w") as f:
        f.write("Hello, {{ project_name }}!")

    # Test normal file generation
    generate_file(project_dir, infile, context, env)

    # Verify file was created and rendered correctly
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile) as f:
        assert f.read() == "Hello, test_project!"

    # Test binary file handling
    binary_file = "test_binary.bin"
    with open(binary_file, "wb") as f:
        f.write(b"binary data")

    generate_file(project_dir, binary_file, context, env)
    binary_outfile = os.path.join(project_dir, binary_file)
    assert os.path.exists(binary_outfile)
    with open(binary_outfile, "rb") as f:
        assert f.read() == b"binary data"

    # Test skip_if_file_exists
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == "Hello, test_project!"  # Should not change

    # Cleanup
    os.remove(infile)
    os.remove(binary_file)
    shutil.rmtree(project_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # First generation
        generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test").exists()

        # Second generation with overwrite
        generate_files(str(repo_dir), output_dir=str(output_dir), overwrite_if_exists=True)
        assert (output_dir / "test").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        # First generation
        generate_files(str(repo_dir), output_dir=str(output_dir))
        original_time = (output_dir / "test" / "file.txt").stat().st_mtime

        # Sleep to ensure different modification time
        time.sleep(0.1)

        # Second generation with skip
        generate_files(str(repo_dir), output_dir=str(output_dir), skip_if_file_exists=True)
        new_time = (output_dir / "test" / "file.txt").stat().st_mtime

        assert original_time == new_time

    # Test with context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "default"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        context = {"project_name": "custom"}
        generate_files(str(repo_dir), context=context, output_dir=str(output_dir))
        assert (output_dir / "custom" / "file.txt").read_text() == "custom"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

        generate_files(str(repo_dir), output_dir=str(output_dir), accept_hooks=True)
        assert (output_dir / "test").exists()

    # Test binary file handling
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Create a binary file
        binary_content = b'\x00\x01\x02\x03'
        (repo_dir / "{{cookiecutter.project_name}}" / "binary.bin").write_bytes(binary_content)

        generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "binary.bin").read_bytes() == binary_content

    # Test copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test", "_copy_without_render": ["*.txt"]}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "file.txt").read_text() == "{{cookiecutter.project_name}}"

    # Test exception handling
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.undefined_var}}")

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(str(repo_dir), output_dir=str(output_dir))

    # Test keep_project_on_failure
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        output_dir = Path(tmpdir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.undefined_var}}")

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(str(repo_dir), output_dir=str(output_dir), keep_project_on_failure=True)

        # Project should still exist
        assert (output_dir / "test").exists()


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path('tests/test-templates/{{cookiecutter.project_name}}')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
            '_copy_without_render': ['*.md'],
        }
    }
    output_dir = Path('tests/output')
    project_dir = output_dir / 'test_project'

    # Clean up if output directory exists
    if project_dir.exists():
        shutil.rmtree(project_dir)

    # Execute
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False,
    )

    # Verify
    assert result == str(project_dir)
    assert project_dir.exists()
    assert (project_dir / 'README.md').exists()
    assert (project_dir / 'src' / 'test_project').exists()

    # Clean up
    shutil.rmtree(project_dir)


# LLM-generated content at query #41
#--------------------------

```python
def test_render_and_create_dir():
    # Test normal directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'test_project'

    # Test directory creation with existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        # Should raise exception when directory exists
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                'test_project',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )

        # Should work when overwrite is enabled
        result_dir, created = render_and_create_dir(
            'test_project',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert result_dir.exists()
        assert not created
        assert result_dir.name == 'test_project'

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'my_project-1.0'


# LLM-generated content at query #42
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
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path))
    assert (tmp_path / "test_project" / "test.txt").exists()
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Hello, test_project!"

    # Test with overwrite_if_exists
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), overwrite_if_exists=True)
    assert (tmp_path / "test_project" / "test.txt").exists()

    # Test with skip_if_file_exists
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), skip_if_file_exists=True)
    assert (tmp_path / "test_project" / "test.txt").exists()

    # Test with extra context
    project_dir = generate_files(
        str(repo_dir),
        output_dir=str(tmp_path),
        extra_context={"project_name": "custom_project"}
    )
    assert (tmp_path / "custom_project" / "test.txt").exists()
    assert (tmp_path / "custom_project" / "test.txt").read_text() == "Hello, custom_project!"

    # Test with accept_hooks=False
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), accept_hooks=False)
    assert (tmp_path / "test_project" / "test.txt").exists()

    # Test with keep_project_on_failure=True
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), keep_project_on_failure=True)
    assert (tmp_path / "test_project" / "test.txt").exists()


# LLM-generated content at query #43
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp')
    env = Environment()

    dir_to_create, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        env
    )
    assert dir_to_create == output_dir / 'test_project'
    assert created is True
    assert dir_to_create.exists()

    # Test with existing directory (should raise exception)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            '{{cookiecutter.project_name}}',
            context,
            output_dir,
            env
        )

    # Test with overwrite_if_exists=True
    dir_to_create, created = render_and_create_dir(
        '{{cookiecutter.project_name}}',
        context,
        output_dir,
        env,
        overwrite_if_exists=True
    )
    assert dir_to_create == output_dir / 'test_project'
    assert created is False  # Directory already existed
    assert dir_to_create.exists()

    # Test empty directory name (should raise exception)
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            '',
            context,
            output_dir,
            env
        )

    # Test with complex template
    context = {'project_name': 'my_project', 'version': '1.0'}
    dir_to_create, created = render_and_create_dir(
        '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
        context,
        output_dir,
        env
    )
    assert dir_to_create == output_dir / 'my_project-1.0'
    assert created is True
    assert dir_to_create.exists()

    # Clean up
    shutil.rmtree(output_dir / 'test_project')
    shutil.rmtree(output_dir / 'my_project-1.0')


# LLM-generated content at query #44
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory creation with existing directory (should raise exception)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / 'existing'
        existing_dir.mkdir()
        context = {'project_name': 'existing'}
        env = Environment()
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test directory creation with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir) / 'existing'
        existing_dir.mkdir()
        context = {'project_name': 'existing'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created  # Directory already existed

    # Test empty directory name (should raise exception)
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test complex template rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {
            'project_name': 'my_project',
            'version': '1.0'
        }
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'my_project-1.0'


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    repo_dir = Path('tests/mocks/basic_template')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = Path('tests/output')
    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'test_project').exists()

    # Test with overwrite_if_exists
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(project_dir).exists()

    # Test with skip_if_file_exists
    context = {'cookiecutter': {'project_name': 'test_project', 'skip_file': 'skip_me'}}
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(project_dir, 'skip_me').exists()

    # Test with hooks
    repo_dir = Path('tests/mocks/template_with_hooks')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'hook_output.txt').exists()

    # Test with keep_project_on_failure
    repo_dir = Path('tests/mocks/template_with_error')
    context = {'cookiecutter': {'project_name': 'test_project'}}
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(project_dir).exists()

    # Test with copy_only_path
    repo_dir = Path('tests/mocks/template_with_copy_only')
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.txt']}}
    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir, 'copy_only.txt').exists()
    with open(Path(project_dir, 'copy_only.txt'), 'r') as f:
        assert f.read() == 'This file should not be rendered'


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_context():
    # Test with a valid context file
    context_file = 'tests/test-data/context.json'
    context = generate_context(context_file)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'Test Project'

    # Test with a non-existent context file
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent_file.json')

    # Test with an invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid_context.json')

    # Test with default_context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra_context
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'

    # Test with both default_context and extra_context
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['project_name'] == 'Extra Project'


# LLM-generated content at query #47
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Hello, test_project!'

    # Test with extra context
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        output_dir = os.path.join(temp_dir, 'output')
        extra_context = {'project_name': 'custom_project'}
        result = generate_files(repo_dir, extra_context=extra_context, output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, 'custom_project', 'test.txt'))
        with open(os.path.join(output_dir, 'custom_project', 'test.txt')) as f:
            assert f.read() == 'Hello, custom_project!'

    # Test overwrite_if_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

        # Try to generate again without overwrite
        with pytest.raises(OutputDirExistsException):
            generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=False)

        # Try to generate again with overwrite
        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

        # Modify the generated file
        with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'w') as f:
            f.write('Modified content')

        # Generate again with skip_if_file_exists
        result = generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Modified content'

    # Test hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, 'hooks'))
        with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('print("Pre hook executed")')
        with open(os.path.join(repo_dir, 'hooks', 'post_gen_project.py'), 'w') as f:
            f.write('print("Post hook executed")')
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)

        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))

    # Test error handling for undefined variables
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.undefined_var}}!')

        output_dir = os.path.join(temp_dir, 'output')
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)


# LLM-generated content at query #48
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)
        dirname = '{{ cookiecutter.project_name }}'
        result, created = render_and_create_dir(dirname, context, output_dir, env)
        assert result == output_dir / 'test_project'
        assert created is True
        assert result.exists()

    # Test directory creation with existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)
        dirname = '{{ cookiecutter.project_name }}'
        result, created = render_and_create_dir(dirname, context, output_dir, env)
        assert result == output_dir / 'test_project'
        assert created is True
        assert result.exists()

        # Try to create the same directory again
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(dirname, context, output_dir, env)

        # Try to create the same directory again with overwrite
        result, created = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        assert result == output_dir / 'test_project'
        assert created is False
        assert result.exists()

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()
        output_dir = Path(tmpdir)
        dirname = '{{ cookiecutter.project_name }}'
        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(dirname, context, output_dir, env)

    # Test undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        output_dir = Path(tmpdir)
        dirname = '{{ cookiecutter.nonexistent_variable }}'
        with pytest.raises(UndefinedVariableInTemplate):
            render_and_create_dir(dirname, context, output_dir, env)


# LLM-generated content at query #49
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/test-data/context.json')
    assert context == {'cookiecutter': {'project_name': 'test'}}

    # Test context generation with default context
    default_context = {'project_name': 'default'}
    context = generate_context('tests/test-data/context.json', default_context)
    assert context == {'cookiecutter': {'project_name': 'default'}}

    # Test context generation with extra context
    extra_context = {'project_name': 'extra'}
    context = generate_context('tests/test-data/context.json', extra_context=extra_context)
    assert context == {'cookiecutter': {'project_name': 'extra'}}

    # Test context generation with invalid JSON
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

    # Test context generation with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-data/nonexistent.json')


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()

    # Create a simple template structure
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("Hello")

    # Test basic generation
    output_dir = tmp_path / "output"
    result = generate_files(
        repo_dir,
        context={"cookiecutter": {"project_name": "generated"}},
        output_dir=output_dir
    )

    assert (output_dir / "generated" / "file.txt").exists()
    assert (output_dir / "generated" / "file.txt").read_text() == "Hello"

    # Test overwrite
    generate_files(
        repo_dir,
        context={"cookiecutter": {"project_name": "generated"}},
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    # Test skip existing
    generate_files(
        repo_dir,
        context={"cookiecutter": {"project_name": "generated"}},
        output_dir=output_dir,
        skip_if_file_exists=True
    )

    # Test hooks
    (repo_dir / "hooks" / "pre_gen_project.py").write_text(
        "import os\nos.makedirs('hook_dir')"
    )
    generate_files(
        repo_dir,
        context={"cookiecutter": {"project_name": "with_hooks"}},
        output_dir=output_dir,
        accept_hooks=True
    )
    assert (output_dir / "with_hooks" / "hook_dir").exists()

    # Test error cases
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir,
            context={"cookiecutter": {}},
            output_dir=output_dir
        )


# LLM-generated content at query #51
#--------------------------

```python
def test_generate_files(tmp_path):
    # Setup test data
    repo_dir = tmp_path / "template"
    repo_dir.mkdir()
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project"}')
    (repo_dir / "{{cookiecutter.project_name}}").mkdir()
    (repo_dir / "{{cookiecutter.project_name}}" / "test.txt").write_text("Hello {{cookiecutter.project_name}}")

    # Test basic file generation
    project_dir = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path,
        overwrite_if_exists=True
    )

    assert (tmp_path / "test_project" / "test.txt").exists()
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Hello test_project"

    # Test with skip_if_file_exists
    generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path,
        skip_if_file_exists=True
    )
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Hello test_project"

    # Test with overwrite_if_exists=False
    with pytest.raises(OutputDirExistsException):
        generate_files(
            repo_dir=str(repo_dir),
            context={"cookiecutter": {"project_name": "test_project"}},
            output_dir=tmp_path,
            overwrite_if_exists=False
        )

    # Test with undefined variable
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir=str(repo_dir),
            context={"cookiecutter": {"project_name": "test_project"}},
            output_dir=tmp_path / "new",
            overwrite_if_exists=True
        )

    # Test with binary file
    binary_file = repo_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project", "_copy_without_render": ["*.bin"]}')

    project_dir = generate_files(
        repo_dir=str(repo_dir),
        context={"cookiecutter": {"project_name": "test_project"}},
        output_dir=tmp_path / "binary_test",
        overwrite_if_exists=True
    )

    assert (tmp_path / "binary_test" / "test_project" / "binary.bin").exists()
    assert (tmp_path / "binary_test" / "test_project" / "binary.bin").read_bytes() == b'\x00\x01\x02\x03'


# LLM-generated content at query #52
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
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path))
    assert (tmp_path / "test_project" / "test.txt").exists()
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Hello, test_project!"

    # Test overwrite_if_exists
    with pytest.raises(OutputDirExistsException):
        generate_files(str(repo_dir), output_dir=str(tmp_path))

    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), overwrite_if_exists=True)
    assert (tmp_path / "test_project" / "test.txt").exists()

    # Test skip_if_file_exists
    (tmp_path / "test_project" / "test.txt").write_text("Existing content")
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path), overwrite_if_exists=True, skip_if_file_exists=True)
    assert (tmp_path / "test_project" / "test.txt").read_text() == "Existing content"

    # Test context override
    project_dir = generate_files(
        str(repo_dir),
        output_dir=str(tmp_path / "override"),
        extra_context={"project_name": "override_project"}
    )
    assert (tmp_path / "override" / "override_project" / "test.txt").read_text() == "Hello, override_project!"

    # Test binary file handling
    binary_file = repo_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    (repo_dir / "cookiecutter.json").write_text('{"project_name": "test_project", "_copy_without_render": ["*.bin"]}')
    project_dir = generate_files(str(repo_dir), output_dir=str(tmp_path / "binary"))
    assert (tmp_path / "binary" / "test_project" / "binary.bin").exists()
    assert (tmp_path / "binary" / "test_project" / "binary.bin").read_bytes() == b"\x00\x01\x02\x03"


# LLM-generated content at query #53
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Hello test_project'

    # Test with overwrite_if_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))

        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(os.path.join(output_dir, 'test_project'))
        with open(os.path.join(output_dir, 'test_project', 'existing.txt'), 'w') as f:
            f.write('existing')

        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        assert os.path.exists(os.path.join(output_dir, 'test_project'))

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(os.path.join(output_dir, 'test_project'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'w') as f:
            f.write('existing')

        result = generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'existing'

    # Test with hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, 'hooks'))
        with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('print("Pre hook executed")')
        with open(os.path.join(repo_dir, 'hooks', 'post_gen_project.py'), 'w') as f:
            f.write('print("Post hook executed")')
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))

        output_dir = os.path.join(temp_dir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)
        assert os.path.exists(os.path.join(output_dir, 'test_project'))

    # Test with keep_project_on_failure
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = os.path.join(temp_dir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.undefined_var}}')

        output_dir = os.path.join(temp_dir, 'output')
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir, keep_project_on_failure=True)
        assert os.path.exists(os.path.join(output_dir, 'test_project'))


# LLM-generated content at query #54
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()

    # Test case 1: Normal directory creation
    dirname = '{{ cookiecutter.project_name }}'
    expected_dir = output_dir / 'test_project'
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    assert result_dir == expected_dir
    assert created is True
    assert result_dir.exists()

    # Test case 2: Directory already exists (should raise exception)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )

    # Test case 3: Overwrite existing directory
    result_dir, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert result_dir == expected_dir
    assert created is False  # Because it already existed

    # Test case 4: Empty directory name (should raise exception)
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, environment)

    # Cleanup
    if expected_dir.exists():
        shutil.rmtree(expected_dir)


# LLM-generated content at query #55
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path('tests/mocks/valid-template')
    output_dir = Path('tests/out')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
            '_copy_without_render': ['*.py']
        }
    }

    # Execute
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True
    )

    # Assert
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()
    assert Path(result, 'test_project', 'README.md').exists()

    # Cleanup
    shutil.rmtree(result)


# LLM-generated content at query #56
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    project_dir = '/tmp/test_project'
    os.makedirs(project_dir, exist_ok=True)

    # Create a test template file
    template_content = "Hello, {{ name }}!"
    template_file = 'test_template.txt'
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template_content)

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Test context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': '\n'
        },
        'name': 'World'
    }

    # Call the function
    generate_file(project_dir, template_file, context, env)

    # Check if file was created and rendered correctly
    expected_file = os.path.join(project_dir, 'test_template.txt')
    assert os.path.exists(expected_file)

    with open(expected_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == "Hello, World!"

    # Cleanup
    os.remove(template_file)
    shutil.rmtree(project_dir)


# LLM-generated content at query #57
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Check results
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with context override
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        # Generate with context override
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            context={"cookiecutter": {"project_name": "override"}},
            overwrite_if_exists=True
        )

        assert (output_dir / "override").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("original")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Modify template
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("modified")

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True,
            skip_if_file_exists=True
        )

        # Should keep original content
        assert (output_dir / "test" / "file.txt").read_text() == "original"

    # Test with hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create template with hooks
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

        # Generate with hooks
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True,
            overwrite_if_exists=True
        )

        assert (output_dir / "test").exists()

    # Test error handling for undefined variables
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create template with undefined variable
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.undefined_var}}").mkdir()

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir,
                overwrite_if_exists=True
            )


# LLM-generated content at query #58
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_path, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_path.exists()
        assert created
        assert dir_path.name == 'test_project'

    # Test directory already exists with overwrite=False
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_path = Path(tmpdir) / 'test_project'
        dir_path.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )

    # Test directory already exists with overwrite=True
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_path = Path(tmpdir) / 'test_project'
        dir_path.mkdir()
        (dir_path / 'existing_file').write_text('content')

        dir_path, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_path.exists()
        assert not created
        assert not (dir_path / 'existing_file').exists()

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        dir_path, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_path.exists()
        assert created
        assert dir_path.name == 'my_project-1.0'


# LLM-generated content at query #59
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "test_template"
    context = {"project_name": "test_project", "author": "Test Author"}
    output_dir = "test_output"
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "{{cookiecutter.project_name}}"), exist_ok=True)

    # Create a simple template file
    with open(os.path.join(repo_dir, "{{cookiecutter.project_name}}", "test.txt"), "w") as f:
        f.write("Hello, {{cookiecutter.author}}!")

    # Test
    result = generate_files(repo_dir, context, output_dir)

    # Assert
    assert os.path.exists(os.path.join(output_dir, "test_project"))
    assert os.path.exists(os.path.join(output_dir, "test_project", "test.txt"))
    with open(os.path.join(output_dir, "test_project", "test.txt"), "r") as f:
        assert f.read() == "Hello, Test Author!"

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #60
#--------------------------

```python
def test_generate_files():
    # Test basic project generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(result, 'test.txt'))
        with open(os.path.join(result, 'test.txt')) as f:
            assert f.read() == 'Hello test'

    # Test with extra context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, extra_context={'project_name': 'custom'}, output_dir=output_dir)
        assert os.path.exists(os.path.join(result, 'test.txt'))
        with open(os.path.join(result, 'test.txt')) as f:
            assert f.read() == 'Hello custom'

    # Test with overwrite_if_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        with open(os.path.join(result, 'test.txt'), 'w') as f:
            f.write('Modified')
        result = generate_files(repo_dir, overwrite_if_exists=True, output_dir=output_dir)
        with open(os.path.join(result, 'test.txt')) as f:
            assert f.read() == 'Hello test'

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        with open(os.path.join(result, 'test.txt'), 'w') as f:
            f.write('Modified')
        result = generate_files(repo_dir, skip_if_file_exists=True, output_dir=output_dir)
        with open(os.path.join(result, 'test.txt')) as f:
            assert f.read() == 'Modified'

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')
        os.makedirs(os.path.join(repo_dir, 'hooks'))
        with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('print("Pre hook executed")')
        with open(os.path.join(repo_dir, 'hooks', 'post_gen_project.py'), 'w') as f:
            f.write('print("Post hook executed")')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, accept_hooks=True, output_dir=output_dir)
        assert os.path.exists(os.path.join(result, 'test.txt'))

    # Test with undefined variable
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello {{cookiecutter.undefined_var}}')

        output_dir = os.path.join(tmpdir, 'output')
        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)

    # Test with binary file
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        binary_file = os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'test binary content')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(result, 'binary.bin'))
        with open(os.path.join(result, 'binary.bin'), 'rb') as f:
            assert f.read() == b'test binary content'

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test', '_copy_without_render': ['*.md']}, f)
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.md'), 'w') as f:
            f.write('Hello {{cookiecutter.project_name}}')

        output_dir = os.path.join(tmpdir, 'output')
        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(result, 'test.md'))
        with open(os.path.join(result, 'test.md')) as f:
            assert f.read() == 'Hello {{cookiecutter.project_name}}'


# LLM-generated content at query #61
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        # Create a simple template
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test_project'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        result = generate_files(repo_dir, output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Hello, test_project!'

    # Test with extra context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'default'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Hello, {{cookiecutter.project_name}}!')

        result = generate_files(
            repo_dir,
            output_dir=output_dir,
            extra_context={'project_name': 'custom'}
        )
        assert os.path.exists(os.path.join(output_dir, 'custom', 'test.txt'))
        with open(os.path.join(output_dir, 'custom', 'test.txt')) as f:
            assert f.read() == 'Hello, custom!'

    # Test overwrite_if_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('First version')

        # First generation
        generate_files(repo_dir, output_dir=output_dir)

        # Modify template
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Second version')

        # Generate again with overwrite
        generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == 'Second version'

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Original content')

        # First generation
        generate_files(repo_dir, output_dir=output_dir)

        # Modify template
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('New content')

        # Generate again with skip
        generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == 'Original content'

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(repo_dir, 'hooks'))
        with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
            f.write('print("Pre hook executed")')

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('Content')

        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)
        assert os.path.exists(os.path.join(output_dir, 'test', 'test.txt'))

    # Test with binary files
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        binary_content = b'\x00\x01\x02\x03'
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'binary.bin'), 'wb') as f:
            f.write(binary_content)

        result = generate_files(repo_dir, output_dir=output_dir)
        with open(os.path.join(output_dir, 'test', 'binary.bin'), 'rb') as f:
            assert f.read() == binary_content

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({
                'project_name': 'test',
                '_copy_without_render': ['*.bin']
            }, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.bin'), 'w') as f:
            f.write('{{cookiecutter.project_name}}')

        result = generate_files(repo_dir, output_dir=output_dir)
        with open(os.path.join(output_dir, 'test', 'test.bin')) as f:
            assert f.read() == '{{cookiecutter.project_name}}'

    # Test error cases
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write('{{undefined_variable}}')

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(repo_dir, output_dir=output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(repo_dir)

        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        os.makedirs(os.path.join(output_dir, 'test'))

        with pytest.raises(OutputDirExistsException):
            generate_files(repo_dir, output_dir=output_dir)


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_generate_file():
    # Setup test environment
    import tempfile
    import os
    from jinja2 import Environment

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = os.path.join(temp_dir, 'project')
        os.makedirs(project_dir)

        # Create a test input file
        infile = 'test_file.txt'
        with open(infile, 'w', encoding='utf-8') as f:
            f.write('Hello, {{ name }}!')

        # Create context
        context = {
            'cookiecutter': {
                '_copy_without_render': [],
                '_new_lines': '\n'
            },
            'name': 'World'
        }

        # Create Jinja2 environment
        env = Environment()

        # Test normal file generation
        generate_file(project_dir, infile, context, env)

        # Verify the output file was created and rendered correctly
        outfile = os.path.join(project_dir, infile)
        assert os.path.exists(outfile)
        with open(outfile, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == 'Hello, World!'

        # Test binary file copy
        binary_file = 'test_binary.bin'
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')

        generate_file(project_dir, binary_file, context, env)

        # Verify binary file was copied without rendering
        binary_outfile = os.path.join(project_dir, binary_file)
        assert os.path.exists(binary_outfile)
        with open(binary_outfile, 'rb') as f:
            binary_content = f.read()
        assert binary_content == b'\x00\x01\x02\x03'

        # Test skip_if_file_exists
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        with open(outfile, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == 'Hello, World!'  # Content should not change


# LLM-generated content at query #64
#--------------------------

```python
def test_generate_files():
    # Setup test data
    repo_dir = Path('tests/test-templates/simple')
    output_dir = Path('tests/test-output')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
        }
    }

    # Call the function
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
    )

    # Assertions
    assert result == str(output_dir / 'test_project')
    assert (output_dir / 'test_project').exists()
    assert (output_dir / 'test_project' / 'README.md').exists()

    # Cleanup
    rmtree(output_dir)


# LLM-generated content at query #65
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
    assert result_path.exists()
    assert created
    assert result_path.name == "test_dir"

    # Test with template rendering
    dirname = "{{ test_var }}"
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path.exists()
    assert created
    assert result_path.name == "test_value"

    # Test overwrite case
    overwrite_if_exists = True
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path.exists()
    assert not created  # Directory already exists

    # Test empty dirname case
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            "", context, output_dir, environment, overwrite_if_exists
        )

    # Test existing directory without overwrite
    overwrite_if_exists = False
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists
        )

    # Cleanup
    if result_path.exists():
        shutil.rmtree(result_path)


# LLM-generated content at query #66
#--------------------------

```python
def test_generate_context():
    # Test basic context generation
    context = generate_context('tests/mocks/cookiecutter.json')
    assert context['cookiecutter']['project_name'] == 'My Project'
    assert context['cookiecutter']['author'] == 'John Doe'

    # Test with default context
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

    # Test with extra context
    extra_context = {'author': 'Jane Smith'}
    context = generate_context(
        'tests/mocks/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['author'] == 'Jane Smith'

    # Test invalid JSON file
    with pytest.raises(ContextDecodingException):
        generate_context('tests/mocks/invalid.json')

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        generate_context('tests/mocks/nonexistent.json')

    # Test with empty context file
    context = generate_context('tests/mocks/empty.json')
    assert context == OrderedDict([])

    # Test with nested context
    context = generate_context('tests/mocks/nested.json')
    assert context['cookiecutter']['nested']['key'] == 'value'


# LLM-generated content at query #67
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


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"test": "value"}
    output_dir = Path("/tmp/test_output")
    environment = Environment()
    overwrite_if_exists = False

    # Test normal case
    result_path, is_created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path.exists()
    assert is_created
    assert result_path == output_dir / dirname

    # Test with template rendering
    dirname = "{{ test }}"
    result_path, is_created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path.exists()
    assert is_created
    assert result_path == output_dir / "value"

    # Test overwrite case
    overwrite_if_exists = True
    result_path, is_created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )
    assert result_path.exists()
    assert not is_created

    # Test empty dirname case
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            "", context, output_dir, environment, overwrite_if_exists
        )

    # Test output dir exists case
    output_dir = Path("/tmp/test_output/existing_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            "test_dir", context, output_dir, environment, overwrite_if_exists=False
        )

    # Cleanup
    shutil.rmtree("/tmp/test_output")


# LLM-generated content at query #70
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "test_template"
    context = {"project_name": "test_project", "author": "Test Author"}
    output_dir = "test_output"

    # Create a test template directory
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "templates"), exist_ok=True)

    # Create a test template file
    with open(os.path.join(repo_dir, "templates", "test.txt"), "w") as f:
        f.write("Hello, {{ cookiecutter.project_name }}!")

    # Create a cookiecutter.json file
    with open(os.path.join(repo_dir, "cookiecutter.json"), "w") as f:
        json.dump({"project_name": "default_project"}, f)

    # Test
    result = generate_files(repo_dir, context, output_dir)

    # Assertions
    assert os.path.exists(result)
    assert os.path.isdir(result)

    # Check if the file was generated correctly
    generated_file = os.path.join(result, "test.txt")
    assert os.path.exists(generated_file)
    with open(generated_file, "r") as f:
        content = f.read()
    assert content == "Hello, test_project!"

    # Cleanup
    shutil.rmtree(repo_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #71
#--------------------------

```python
def test_render_and_create_dir():
    # Setup
    dirname = "test_dir"
    context = {"project_name": "test_project"}
    output_dir = Path("/tmp")
    environment = Environment()

    # Test successful directory creation
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment
    )
    assert result_path.exists()
    assert created
    assert result_path.name == dirname

    # Test directory already exists without overwrite
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=False
        )

    # Test directory already exists with overwrite
    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists=True
    )
    assert result_path.exists()
    assert not created

    # Test empty directory name
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

    # Test template rendering in directory name
    dirname_template = "{{ cookiecutter.project_name }}"
    result_path, created = render_and_create_dir(
        dirname_template, context, output_dir, environment
    )
    assert result_path.name == "test_project"

    # Cleanup
    shutil.rmtree(result_path)


# LLM-generated content at query #72
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = "tests/mocks/valid-template"
    output_dir = "tests/mocks/output"
    context = {"cookiecutter": {"project_name": "test_project"}}
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    # Verify project directory was created
    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

    # Verify files were generated
    expected_files = [
        os.path.join(project_dir, "README.md"),
        os.path.join(project_dir, "setup.py"),
        os.path.join(project_dir, "src", "test_project", "__init__.py")
    ]
    for file_path in expected_files:
        assert os.path.exists(file_path)

    # Verify content was rendered
    with open(os.path.join(project_dir, "README.md")) as f:
        content = f.read()
        assert "test_project" in content

    # Cleanup
    shutil.rmtree(project_dir)


# LLM-generated content at query #73
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


# LLM-generated content at query #74
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

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

        # Check results
        assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
        with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
            assert f.read() == 'Hello test_project!'

    # Test with extra context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'default'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files with extra context
        extra_context = {'project_name': 'custom'}
        result = generate_files(repo_dir, extra_context=extra_context, output_dir=output_dir)

        # Check results
        assert os.path.exists(os.path.join(output_dir, 'custom', 'test.txt'))
        with open(os.path.join(output_dir, 'custom', 'test.txt')) as f:
            assert f.read() == 'Hello custom!'

    # Test with copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a template with copy_without_render
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({
                'project_name': 'test',
                '_copy_without_render': ['*.txt']
            }, f)

        # Create a template file that should be copied without rendering
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files
        result = generate_files(repo_dir, output_dir=output_dir)

        # Check that the file was copied without rendering
        assert os.path.exists(os.path.join(output_dir, 'test', 'test.txt'))
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == template_content

    # Test with overwrite_if_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files first time
        result = generate_files(repo_dir, output_dir=output_dir)

        # Try to generate again with overwrite_if_exists=True
        result = generate_files(repo_dir, output_dir=output_dir, overwrite_if_exists=True)

        # Check results
        assert os.path.exists(os.path.join(output_dir, 'test', 'test.txt'))
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == 'Hello test!'

    # Test with skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a simple template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files first time
        result = generate_files(repo_dir, output_dir=output_dir)

        # Modify the generated file
        with open(os.path.join(output_dir, 'test', 'test.txt'), 'w') as f:
            f.write('Modified content')

        # Try to generate again with skip_if_file_exists=True
        result = generate_files(repo_dir, output_dir=output_dir, skip_if_file_exists=True)

        # Check that the file was not overwritten
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == 'Modified content'

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a template with hooks
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        # Create a pre hook
        hook_dir = os.path.join(repo_dir, 'hooks')
        os.makedirs(hook_dir)
        with open(os.path.join(hook_dir, 'pre_gen_project.py'), 'w') as f:
            f.write('print("Pre hook executed")')

        # Create a post hook
        with open(os.path.join(hook_dir, 'post_gen_project.py'), 'w') as f:
            f.write('print("Post hook executed")')

        # Create a template file
        template_content = 'Hello {{ cookiecutter.project_name }}!'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'test.txt'), 'w') as f:
            f.write(template_content)

        # Generate files with hooks
        result = generate_files(repo_dir, output_dir=output_dir, accept_hooks=True)

        # Check results
        assert os.path.exists(os.path.join(output_dir, 'test', 'test.txt'))
        with open(os.path.join(output_dir, 'test', 'test.txt')) as f:
            assert f.read() == 'Hello test!'

    # Test with binary files
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, 'output')

        # Create a template
        os.makedirs(repo_dir)
        with open(os.path.join(repo_dir, 'cookiecutter.json'), 'w') as f:
            json.dump({'project_name': 'test'}, f)

        # Create a binary file
        binary_content = b'\x00\x01\x02\x03'
        os.makedirs(os.path.join(repo_dir, '{{cookiecutter.project_name}}'))
        with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}', 'binary.bin'), 'wb') as f:
            f.write(binary_content)

        # Generate files
        result = generate_files(repo_dir, output_dir=output_dir)

        # Check that the binary file was copied correctly
        assert os.path.exists(os.path.join(output_dir, 'test', 'binary.bin'))
        with open(os.path.join(output_dir, 'test', 'binary.bin'), 'rb') as f:
            assert f.read() == binary_content

    # Test with undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, 'template')
        output_dir = os.path.join(tmpdir, '


# LLM-generated content at query #75
#--------------------------

```python
def test_generate_files():
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple template structure
        template_dir = Path(temp_dir) / "template"
        template_dir.mkdir()

        # Create a cookiecutter.json file
        context_file = template_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "test_project", "project_slug": "{{ cookiecutter.project_name.lower().replace(\' \', \'_\') }}"}')

        # Create a simple template file
        template_file = template_dir / "{{ cookiecutter.project_slug }}.txt"
        template_file.write_text("Hello, {{ cookiecutter.project_name }}!")

        # Create a copy-only file
        copy_file = template_dir / "copy_only.txt"
        copy_file.write_text("This file should be copied without rendering.")

        # Create a copy-only directory
        copy_dir = template_dir / "copy_only_dir"
        copy_dir.mkdir()
        copy_dir_file = copy_dir / "file.txt"
        copy_dir_file.write_text("This file is in a copy-only directory.")

        # Create a context with copy-only paths
        context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "_copy_without_render": ["copy_only.txt", "copy_only_dir/*"]
            }
        }

        # Create an output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        # Test
        result = generate_files(
            repo_dir=str(template_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )

        # Assertions
        # Check that the project directory was created
        assert Path(result).exists()

        # Check that the rendered file was created with the correct content
        rendered_file = Path(result) / "test_project.txt"
        assert rendered_file.exists()
        assert rendered_file.read_text() == "Hello, Test Project!"

        # Check that the copy-only file was copied without rendering
        copy_only_file = Path(result) / "copy_only.txt"
        assert copy_only_file.exists()
        assert copy_only_file.read_text() == "This file should be copied without rendering."

        # Check that the copy-only directory was copied without rendering
        copy_only_dir = Path(result) / "copy_only_dir"
        assert copy_only_dir.exists()
        copy_only_dir_file = copy_only_dir / "file.txt"
        assert copy_only_dir_file.exists()
        assert copy_only_dir_file.read_text() == "This file is in a copy-only directory."

        # Check that the project directory name was rendered correctly
        assert Path(result).name == "test_project"


# LLM-generated content at query #76
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        (output_dir / "test").mkdir()

        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True
        )
        assert (output_dir / "test").exists()

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        (output_dir / "test").mkdir()
        (output_dir / "test" / "file.txt").write_text("existing")

        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            skip_if_file_exists=True
        )
        assert (output_dir / "test" / "file.txt").read_text() == "existing"

    # Test with hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post')")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(
            str(repo_dir),
            output_dir=str(output_dir),
            accept_hooks=True
        )
        assert (output_dir / "test").exists()

    # Test with context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "default"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()

        output_dir = Path(tmpdir) / "output"
        context = {"project_name": "custom"}
        result = generate_files(
            str(repo_dir),
            context=context,
            output_dir=str(output_dir)
        )
        assert (output_dir / "custom").exists()

    # Test with binary files
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        binary_content = b"\x00\x01\x02\x03"
        (repo_dir / "{{cookiecutter.project_name}}" / "binary.bin").write_bytes(binary_content)

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "binary.bin").read_bytes() == binary_content

    # Test with _copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test", "_copy_without_render": ["*.md"]}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "readme.md").write_text("# {{cookiecutter.project_name}}")

        output_dir = Path(tmpdir) / "output"
        result = generate_files(str(repo_dir), output_dir=str(output_dir))
        assert (output_dir / "test" / "readme.md").read_text() == "# {{cookiecutter.project_name}}"


# LLM-generated content at query #77
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'test_project'

    # Test directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )

    # Test directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert dir_to_create.exists()
        assert not created  # Directory existed before
        assert dir_to_create.name == 'test_project'

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test template rendering in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'my_project', 'version': '1.0'}
        env = Environment()
        dir_to_create, created = render_and_create_dir(
            '{{cookiecutter.project_name}}-{{cookiecutter.version}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert dir_to_create.exists()
        assert created
        assert dir_to_create.name == 'my_project-1.0'


# LLM-generated content at query #78
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        # Create a simple template
        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        assert (output_dir / "test" / "file.txt").exists()
        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test with context override
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.project_name}}")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            context={"cookiecutter": {"project_name": "override"}},
            overwrite_if_exists=True
        )

        assert (output_dir / "override" / "file.txt").exists()
        assert (output_dir / "override" / "file.txt").read_text() == "override"

    # Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("content")

        # First generation
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            overwrite_if_exists=True
        )

        # Second generation with skip
        generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            skip_if_file_exists=True
        )

        assert (output_dir / "test" / "file.txt").read_text() == "content"

    # Test hooks
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "hooks" / "pre_gen_project.py").write_text("print('pre hook')")
        (repo_dir / "hooks" / "post_gen_project.py").write_text("print('post hook')")

        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=output_dir,
            accept_hooks=True,
            overwrite_if_exists=True
        )

        assert (output_dir / "test").exists()

    # Test error cases
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "template"
        output_dir = Path(temp_dir) / "output"

        repo_dir.mkdir()
        (repo_dir / "cookiecutter.json").write_text('{"project_name": "test"}')
        (repo_dir / "{{cookiecutter.project_name}}").mkdir()
        (repo_dir / "{{cookiecutter.project_name}}" / "file.txt").write_text("{{cookiecutter.undefined_var}}")

        with pytest.raises(UndefinedVariableInTemplate):
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=output_dir,
                overwrite_if_exists=True
            )


# LLM-generated content at query #79
#--------------------------

```python
def test_generate_file(tmp_path):
    # Setup test environment
    project_dir = str(tmp_path / "project")
    os.makedirs(project_dir)

    # Create a test template file
    template_content = "Hello, {{ name }}!"
    template_file = tmp_path / "template.txt"
    template_file.write_text(template_content)

    # Create context
    context = {
        'cookiecutter': {
            '_copy_without_render': [],
            '_new_lines': None
        },
        'name': 'World'
    }

    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(tmp_path)))

    # Test file generation
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    # Verify the output file
    output_file = Path(project_dir) / "template.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello, World!"

    # Test binary file handling
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03')
    generate_file(
        project_dir=project_dir,
        infile=str(binary_file),
        context=context,
        env=env,
        skip_if_file_exists=False
    )
    output_binary = Path(project_dir) / "binary.bin"
    assert output_binary.exists()
    assert output_binary.read_bytes() == b'\x00\x01\x02\x03'

    # Test skip_if_file_exists
    generate_file(
        project_dir=project_dir,
        infile=str(template_file),
        context=context,
        env=env,
        skip_if_file_exists=True
    )
    assert output_file.read_text() == "Hello, World!"  # Should remain unchanged

    # Test empty directory name
    empty_dir = tmp_path / "empty_dir"
    os.makedirs(empty_dir)
    with pytest.raises(EmptyDirNameException):
        generate_file(
            project_dir=str(empty_dir),
            infile="",
            context=context,
            env=env,
            skip_if_file_exists=False
        )


# LLM-generated content at query #80
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
        generate_context('non_existent_file.json')


# LLM-generated content at query #81
#--------------------------

```python
def test_generate_files():
    # Setup
    repo_dir = Path("tests/test-templates/test-template")
    output_dir = Path("tests/test-output")
    context = {
        "project_name": "test_project",
        "project_slug": "test_project",
        "author": "Test Author",
    }

    # Clean up output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Execute
    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False,
    )

    # Verify
    assert result == str(output_dir / "test_project")
    assert (output_dir / "test_project").exists()
    assert (output_dir / "test_project" / "README.md").exists()
    assert (output_dir / "test_project" / "setup.py").exists()

    # Clean up
    shutil.rmtree(output_dir)


# LLM-generated content at query #82
#--------------------------

```python
def test_generate_files():
    # Test basic file generation
    repo_dir = Path('tests/fake-repo-pre')
    output_dir = Path('tests/output')
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project',
        }
    }

    project_dir = generate_files(repo_dir, context, output_dir)
    assert Path(project_dir).exists()
    assert Path(project_dir, 'test_project').exists()

    # Test with overwrite_if_exists=True
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(project_dir).exists()

    # Test with skip_if_file_exists=True
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(project_dir).exists()

    # Test with accept_hooks=False
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert Path(project_dir).exists()

    # Test with keep_project_on_failure=True
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(project_dir).exists()

    # Test with invalid context
    invalid_context = {
        'cookiecutter': {
            'invalid_variable': 'test',
        }
    }
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, invalid_context, output_dir)

    # Test with non-existent repo_dir
    with pytest.raises(FileNotFoundError):
        generate_files(Path('non-existent-repo'), context, output_dir)


# LLM-generated content at query #83
#--------------------------

```python
def test_generate_context():
    # Test with valid context file
    context_file = 'tests/test_context.json'
    default_context = {'project_name': 'test_project'}
    extra_context = {'project_slug': 'test_slug'}

    context = generate_context(context_file, default_context, extra_context)

    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'test_project'
    assert context['cookiecutter']['project_slug'] == 'test_slug'

    # Test with invalid context file
    invalid_context_file = 'tests/invalid_context.json'
    with pytest.raises(ContextDecodingException):
        generate_context(invalid_context_file)

    # Test with missing context file
    missing_context_file = 'tests/missing_context.json'
    with pytest.raises(FileNotFoundError):
        generate_context(missing_context_file)

    # Test with empty context file
    empty_context_file = 'tests/empty_context.json'
    with pytest.raises(ContextDecodingException):
        generate_context(empty_context_file)

    # Test with no default or extra context
    context = generate_context(context_file)
    assert 'cookiecutter' in context
    assert context['cookiecutter']['project_name'] == 'default_project'


# LLM-generated content at query #84
#--------------------------

```python
def test_render_and_create_dir():
    # Test basic directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'test_project'

    # Test directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()

        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env,
                overwrite_if_exists=False
            )

    # Test directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project'}
        env = Environment()
        existing_dir = Path(tmpdir) / 'test_project'
        existing_dir.mkdir()
        test_file = existing_dir / 'test.txt'
        test_file.write_text('old content')

        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}',
            {'cookiecutter': context},
            tmpdir,
            env,
            overwrite_if_exists=True
        )
        assert result_dir.exists()
        assert not created
        assert result_dir.name == 'test_project'
        assert not test_file.exists()

    # Test empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': ''}
        env = Environment()

        with pytest.raises(EmptyDirNameException):
            render_and_create_dir(
                '{{cookiecutter.project_name}}',
                {'cookiecutter': context},
                tmpdir,
                env
            )

    # Test nested directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'project_name': 'test_project', 'nested': 'nested_dir'}
        env = Environment()
        result_dir, created = render_and_create_dir(
            '{{cookiecutter.project_name}}/{{cookiecutter.nested}}',
            {'cookiecutter': context},
            tmpdir,
            env
        )
        assert result_dir.exists()
        assert created
        assert result_dir.name == 'nested_dir'
        assert result_dir.parent.name == 'test_project'


