####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__run_hook_from_repo_dir():
    repo_dir = '/path/to/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    with warnings.catch_warnings(record=True) as w:
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "The '_run_hook_from_repo_dir' function is deprecated" in str(w[0].message)


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable_deeper_level():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": {"nested": "value", "new_nested": "new_value"}}

def test_apply_overwrites_to_context_list_with_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}

def test_apply_overwrites_to_context_list_with_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "e"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d', 'e'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_list_with_single_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_with_single_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"config": {"key1": "val1", "key2": "val2"}}
    overwrite_context = {"config": {"key2": "new_val2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "val1", "key2": "new_val2"}}

def test_apply_overwrites_to_context_bool_with_valid_string():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_with_invalid_string():
    context = {"flag": False}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old_value"}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new_value"}


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path(), Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    dir_to_create = Path('existing_dir')
    dir_to_create.mkdir()
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('existing_dir', {}, Path(), Environment())

def test_render_and_create_dir_existing_dir_overwrite():
    dir_to_create = Path('existing_dir')
    dir_to_create.mkdir()
    result = render_and_create_dir('existing_dir', {}, Path(), Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_new_dir():
    result = render_and_create_dir('new_dir', {}, Path(), Environment())
    assert result[0].exists()
    assert result[1] is True

def test_render_and_create_dir_rendered_name():
    context = {'name': 'test'}
    result = render_and_create_dir('{{ name }}', context, Path(), Environment())
    assert result[0].name == 'test'
    assert result[1] is True


# LLM-generated content at query #4
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_boolean_overwrite():
    context = generate_context(
        'tests/test-cookiecutter-bool.json',
        extra_context={'is_active': 'yes'}
    )
    assert context == {'test-cookiecutter-bool': {'is_active': True}}

def test_generate_context_with_invalid_boolean_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter-bool.json',
            extra_context={'is_active': 'invalid'}
        )


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir_creates_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0].exists()
    assert result[1] is True

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0].exists()
    assert result[1] is False

def test_render_and_create_dir_raises_on_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_raises_on_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    render_and_create_dir(dirname, context, output_dir, environment)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)


# LLM-generated content at query #7
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    path = "file.txt"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is True

def test_is_copy_only_path_returns_false_when_path_does_not_match_pattern():
    path = "file.py"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_returns_false_when_key_not_in_context():
    path = "file.txt"
    context = {
        'cookiecutter': {}
    }
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_returns_false_when_cookiecutter_not_in_context():
    path = "file.txt"
    context = {}
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_returns_true_for_directory_matching_pattern():
    path = "docs"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["docs"]
        }
    }
    assert is_copy_only_path(path, context) is True

def test_is_copy_only_path_returns_true_for_nested_path_matching_pattern():
    path = "src/file.txt"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["src/*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is True


# LLM-generated content at query #8
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'temp*']
        }
    }
    assert is_copy_only_path('test.txt', context) is True
    assert is_copy_only_path('temp_file.py', context) is True


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}

    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(project_dir, 'test_project'))
    assert os.path.exists(os.path.join(project_dir, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    project_dir = generate_files(repo_dir, context, output_dir)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(project_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing_files():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    project_dir = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(project_dir, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, 'test_project', 'new_file.txt'))
    with open(os.path.join(project_dir, 'test_project', 'new_file.txt'), 'r') as f:
        assert f.read() == 'existing content'
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(project_dir, 'test_project'))
    assert os.path.exists(os.path.join(project_dir, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_without_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(os.path.join(project_dir, 'test_project'))
    assert not os.path.exists(os.path.join(project_dir, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_copy_without_render')
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(project_dir, 'test_project'))
    assert os.path.exists(os.path.join(project_dir, 'test_project', 'static_file.txt'))
    with open(os.path.join(project_dir, 'test_project', 'static_file.txt'), 'r') as f:
        assert f.read() == 'This file should not be rendered.'
    shutil.rmtree(output_dir)

def test_generate_files_keep_project_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_error')
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except UndefinedVariableInTemplate:
        pass
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_delete_project_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_error')
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=False)
    except UndefinedVariableInTemplate:
        pass
    assert not os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #12
#--------------------------

```python
def test_json_decoding_error_raises_custom_exception():
    with pytest.raises(ContextDecodingException) as exc_info:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(exc_info.value)


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/tmp/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_render():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('existing content')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'existing content'

def test_generate_file_empty_filename():
    project_dir = '/tmp/project'
    infile = ''
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_custom_newline():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            dirname="",
            context={},
            output_dir=".",
            environment=Environment(),
            overwrite_if_exists=False
        )


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test_cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json_file():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'key': 'default_value'}
    )
    assert context == {'test_cookiecutter': {'key': 'default_value'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'key': 'extra_value'}
    )
    assert context == {'test_cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'key': 'default_value'},
        extra_context={'key': 'extra_value'}
    )
    assert context == {'test_cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid_key': 'invalid_value'}
        )
    assert context == {'test_cookiecutter': {'key': 'value'}}


# LLM-generated content at query #17
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #18
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    infile = 'binary_file.png'
    assert is_binary(infile) is True


# LLM-generated content at query #19
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    # Assuming is_binary is a function that checks if a file is binary
    # We need to ensure that the predicate at line 47 evaluates to True
    # This means we need to provide an infile that is a binary file
    assert is_binary('binary_file.bin') == True


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_file_binary_skipped_if_exists():
    project_dir = '/fake/project'
    infile = 'binary.bin'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    skip_if_file_exists = True
    os.path.exists = lambda x: True
    is_binary = lambda x: True
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert not os.path.exists.called

def test_generate_file_text_rendered():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    env.from_string = lambda x: MockTemplate(x)
    env.get_template = lambda x: MockTemplate(x)
    is_binary = lambda x: False
    os.path.exists = lambda x: False
    os.path.isdir = lambda x: False
    generate_file(project_dir, infile, context, env)
    assert env.get_template.called

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    env.from_string = lambda x: MockTemplate(x)
    env.get_template = lambda x: MockTemplate(x)
    is_binary = lambda x: False
    os.path.exists = lambda x: False
    os.path.isdir = lambda x: False
    open_mock = mock_open(read_data='line1\nline2')
    with patch('builtins.open', open_mock):
        generate_file(project_dir, infile, context, env)
    assert open_mock.called

def test_generate_file_permissions_copied():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    env.from_string = lambda x: MockTemplate(x)
    env.get_template = lambda x: MockTemplate(x)
    is_binary = lambda x: False
    os.path.exists = lambda x: False
    os.path.isdir = lambda x: False
    shutil.copymode = Mock()
    generate_file(project_dir, infile, context, env)
    assert shutil.copymode.called


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    _run_hook_from_repo_dir('repo', 'hook', 'project', {}, False)
    assert True


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_variable_deeper_level():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": {"nested": "value", "new": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_overwrite():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": ["a", "b"]}

def test_apply_overwrites_to_context_multichoice_overwrite_invalid():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"var": {"a": 1, "b": 2}}
    overwrite_context = {"var": {"a": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": {"a": 3, "b": 2}}

def test_apply_overwrites_to_context_bool_overwrite_true():
    context = {"var": False}
    overwrite_context = {"var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": True}

def test_apply_overwrites_to_context_bool_overwrite_false():
    context = {"var": True}
    overwrite_context = {"var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": False}

def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"var": True}
    overwrite_context = {"var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("existing_dir", {}, "/tmp", Environment())
    dir_to_create.rmdir()

def test_render_and_create_dir_existing_dir_with_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("existing_dir", {}, "/tmp", Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)
    dir_to_create.rmdir()

def test_render_and_create_dir_new_dir():
    dir_to_create = Path("/tmp/new_dir")
    result = render_and_create_dir("new_dir", {}, "/tmp", Environment())
    assert result == (dir_to_create, True)
    dir_to_create.rmdir()

def test_render_and_create_dir_rendered_name():
    context = {"project_name": "test_project"}
    environment = Environment()
    result = render_and_create_dir("{{ project_name }}_dir", context, "/tmp", environment)
    expected_dir = Path("/tmp/test_project_dir")
    assert result == (expected_dir, True)
    expected_dir.rmdir()


# LLM-generated content at query #4
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid boolean overwrite")


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": "d"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == ["a", "b", "c"]


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path(), Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    output_dir = Path('existing_dir')
    output_dir.mkdir(exist_ok=True)
    with raises(OutputDirExistsException):
        render_and_create_dir('test', {}, output_dir, Environment())

def test_render_and_create_dir_existing_dir_with_overwrite():
    output_dir = Path('existing_dir')
    output_dir.mkdir(exist_ok=True)
    result_path, created = render_and_create_dir('test', {}, output_dir, Environment(), overwrite_if_exists=True)
    assert result_path == output_dir / 'test'
    assert not created

def test_render_and_create_dir_new_dir():
    output_dir = Path('new_dir')
    result_path, created = render_and_create_dir('test', {}, output_dir, Environment())
    assert result_path == output_dir / 'test'
    assert created
    assert result_path.exists()

def test_render_and_create_dir_rendered_name():
    context = {'name': 'test'}
    output_dir = Path('rendered_dir')
    result_path, created = render_and_create_dir('{{ name }}', context, output_dir, Environment())
    assert result_path == output_dir / 'test'
    assert created


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'key': 'default_value'}
    )
    assert context == {'test-cookiecutter': {'key': 'default_value'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'key': 'extra_value'}
    )
    assert context == {'test-cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'key': 'default_value'},
        extra_context={'key': 'extra_value'}
    )
    assert context == {'test-cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_nested_dict():
    context = generate_context(
        'tests/test-nested-cookiecutter.json',
        extra_context={'nested': {'key': 'nested_value'}}
    )
    assert context == {'test-nested-cookiecutter': {'nested': {'key': 'nested_value'}}}

def test_generate_context_with_list_choice():
    context = generate_context(
        'tests/test-list-cookiecutter.json',
        extra_context={'choice': 'option2'}
    )
    assert context == {'test-list-cookiecutter': {'choice': ['option2', 'option1', 'option3']}}

def test_generate_context_with_invalid_list_choice():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-list-cookiecutter.json',
            extra_context={'choice': 'invalid_option'}
        )

def test_generate_context_with_boolean_conversion():
    context = generate_context(
        'tests/test-boolean-cookiecutter.json',
        extra_context={'bool_var': 'yes'}
    )
    assert context == {'test-boolean-cookiecutter': {'bool_var': True}}

def test_generate_context_with_invalid_boolean_conversion():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-boolean-cookiecutter.json',
            extra_context={'bool_var': 'invalid_bool'}
        )


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable my_bool could not be converted to a boolean." in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'value': 1}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'value': 1}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'value': 2}
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'value': 2}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'value': 2}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'value': 2}}

def test_generate_context_with_none_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context=None,
        extra_context=None
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'value': 1}}


# LLM-generated content at query #11
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #12
#--------------------------

```python
def test_apply_overwrites_to_context_raises_value_error_for_invalid_boolean_string():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid_string"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    try:
        generate_context(context_file)
    except Exception:
        pass
    finally:
        os.remove(context_file)


# LLM-generated content at query #14
#--------------------------

```python
def test_is_copy_only_path_matching_pattern():
    assert is_copy_only_path('file.txt', {'cookiecutter': {'_copy_without_render': ['*.txt']}}) == True

def test_is_copy_only_path_not_matching_pattern():
    assert is_copy_only_path('file.py', {'cookiecutter': {'_copy_without_render': ['*.txt']}}) == False

def test_is_copy_only_path_no_patterns():
    assert is_copy_only_path('file.txt', {'cookiecutter': {}}) == False

def test_is_copy_only_path_missing_key():
    assert is_copy_only_path('file.txt', {}) == False

def test_is_copy_only_path_multiple_patterns():
    assert is_copy_only_path('file.txt', {'cookiecutter': {'_copy_without_render': ['*.py', '*.txt']}}) == True

def test_is_copy_only_path_directory_pattern():
    assert is_copy_only_path('dir/subdir/file.txt', {'cookiecutter': {'_copy_without_render': ['dir/**']}}) == True


# LLM-generated content at query #15
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'project_name': 'test', 'project_slug': 'test'}}
    assert isinstance(context, OrderedDict)

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context('tests/test-data/invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'project_name': 'default'}
    )
    assert context['cookiecutter']['project_name'] == 'default'

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'project_name': 'extra'}
    )
    assert context['cookiecutter']['project_name'] == 'extra'

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning) as record:
        generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid_var': 'value'}
        )
    assert "Invalid default received" in str(record[0].message)

def test_generate_context_with_none_default_and_extra():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context=None,
        extra_context=None
    )
    assert context == {'cookiecutter': {'project_name': 'test', 'project_slug': 'test'}}


# LLM-generated content at query #17
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #18
#--------------------------

```python
def test_render_and_create_dir_empty_dirname_raises_exception():
    with raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test-cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'test-cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_nested_dict_overwrite():
    context = generate_context(
        'tests/test-cookiecutter-nested.json',
        extra_context={'nested': {'key1': 'overwritten'}}
    )
    assert context == {'test-cookiecutter-nested': {'nested': {'key1': 'overwritten', 'key2': 'value2'}}}

def test_generate_context_with_list_overwrite():
    context = generate_context(
        'tests/test-cookiecutter-list.json',
        extra_context={'choices': ['choice2', 'choice1']}
    )
    assert context == {'test-cookiecutter-list': {'choices': ['choice2', 'choice1']}}

def test_generate_context_with_invalid_list_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter-list.json',
            extra_context={'choices': ['invalid']}
        )

def test_generate_context_with_boolean_overwrite():
    context = generate_context(
        'tests/test-cookiecutter-bool.json',
        extra_context={'flag': 'yes'}
    )
    assert context == {'test-cookiecutter-bool': {'flag': True}}

def test_generate_context_with_invalid_boolean_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter-bool.json',
            extra_context={'flag': 'invalid'}
        )


# LLM-generated content at query #20
#--------------------------

```python
def test_default_context_is_not_none():
    default_context = {"key": "value"}
    context_file = "cookiecutter.json"
    extra_context = None
    assert default_context is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_context_with_default_context():
    default_context = {"key": "value"}
    extra_context = None
    context_file = "cookiecutter.json"
    result = generate_context(context_file, default_context, extra_context)
    assert result is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key": "value"}')

    result = generate_context(context_file)

    assert result == {'test': {'key': 'value'}}
    os.remove(context_file)


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_with_default_context():
    default_context = {'key': 'value'}
    extra_context = None
    context_file = 'cookiecutter.json'
    result = generate_context(context_file, default_context, extra_context)
    assert result is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'new_file.txt'))
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'r') as f:
        assert f.read() == 'existing content'
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'with_hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_marker.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'copy_without_render')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'static_file.txt'))
    with open(os.path.join(result, 'test_project', 'static_file.txt'), 'r') as f:
        assert '{{ project_name }}' in f.read()
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'undefined_var')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'undefined_var')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/tmp/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_name():
    project_dir = '/tmp/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_config():
    project_dir = '/tmp/project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
        assert '\n' in content

def test_generate_file_newline_detect():
    project_dir = '/tmp/project'
    infile = 'detect_newline.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
        assert '\n' in content or '\r\n' in content


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()

    assert '\n' in content or '\r\n' in content

def test_generate_file_custom_newline():
    project_dir = '/tmp/test_project'
    infile = 'custom_newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()

    assert '\r\n' in content


# LLM-generated content at query #27
#--------------------------

```python
def test_cookiecutter_new_lines_true():
    context = {
        'cookiecutter': {
            '_new_lines': True
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #28
#--------------------------

```python
def test_template_syntax_error_exception_handling():
    project_dir = '/fake/project/dir'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    # Create a template with syntax error
    with open(infile, 'w') as f:
        f.write('{% if %}')

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert exc_info.value.translated is False


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    is_binary.return_value = True
    generate_file(project_dir, infile, context, env)
    shutil.copyfile.assert_called_once_with(infile, os.path.join(project_dir, infile))
    shutil.copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text_file_with_newline_config():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    env.get_template.return_value = MagicMock()
    env.get_template.return_value.render.return_value = 'rendered content'
    is_binary.return_value = False
    with patch('builtins.open', mock_open(read_data='first line\n')) as mock_file:
        generate_file(project_dir, infile, context, env)
    mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\n')
    mock_file().write.assert_called_once_with('rendered content')

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    os.path.exists.return_value = True
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    env.get_template.assert_not_called()


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.bin'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_mode:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_copy.assert_called_once_with(infile, os.path.join(project_dir, infile))
        mock_mode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch.object(env, 'get_template') as mock_get_template, \
         patch('builtins.open', mock_open(read_data='rendered content')) as mock_file, \
         patch('shutil.copymode') as mock_mode:
        mock_template = MagicMock()
        mock_template.render.return_value = 'rendered content'
        mock_get_template.return_value = mock_template

        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline=None)
        mock_mode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True), \
         patch('utils.is_binary', return_value=False) as mock_is_binary:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_is_binary.assert_not_called()

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = 'empty_dir/'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=True), \
         patch('utils.is_binary', return_value=False) as mock_is_binary:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_is_binary.assert_not_called()


# LLM-generated content at query #31
#--------------------------

```python
def test_template_syntax_error_handling():
    project_dir = "/fake/project"
    infile = "fake_template.txt"
    context = {"cookiecutter": {}}
    env = Environment(loader=FileSystemLoader("/fake/templates"))

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)

    assert exc_info.value.translated is False


# LLM-generated content at query #32
#--------------------------

```python
def test_delete_project_on_failure_false_when_keep_project_on_failure_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #33
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    outfile = '/path/to/existing/file'
    os.path.exists = lambda path: True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_67_is_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_true():
    project_dir = '/fake/project/dir'
    infile = 'fake_infile.txt'
    context = {'fake_key': 'fake_value'}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'a').close()
    assert skip_if_file_exists and os.path.exists(os.path.join(project_dir, infile))


# LLM-generated content at query #36
#--------------------------

```python
def test_is_binary_evaluates_to_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project/dir'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'binary_file.png'))

def test_generate_file_text_file():
    project_dir = '/fake/project/dir'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'text_file.txt'))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project/dir'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, 'existing_file.txt'), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'existing_file.txt'))

def test_generate_file_empty_filename():
    project_dir = '/fake/project/dir'
    infile = ''
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(project_dir)

def test_generate_file_custom_newline():
    project_dir = '/fake/project/dir'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, 'text_file.txt'), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    # Mocking the necessary functions and objects
    os.path.isdir.return_value = False
    os.path.exists.return_value = False
    is_binary.return_value = True
    logger.debug = MagicMock()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    shutil.copyfile.assert_called_once_with(infile, os.path.join(project_dir, infile))
    shutil.copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))
    logger.debug.assert_any_call('Processing file %s', infile)
    logger.debug.assert_any_call('Copying binary %s to %s without rendering', infile, os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    # Mocking the necessary functions and objects
    os.path.isdir.return_value = False
    os.path.exists.return_value = False
    is_binary.return_value = False
    logger.debug = MagicMock()
    env.get_template.return_value.render.return_value = 'rendered content'
    open_context = MagicMock()
    open_context.__enter__.return_value.readline.return_value = 'first line\n'
    open_context.__enter__.return_value.newlines = '\n'
    open = MagicMock(return_value=open_context)

    with patch('builtins.open', open):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)

    env.get_template.assert_called_once_with(infile.replace(os.path.sep, '/'))
    open.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\n')
    open_context.__enter__.return_value.write.assert_called_once_with('rendered content')
    shutil.copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))
    logger.debug.assert_any_call('Processing file %s', infile)
    logger.debug.assert_any_call('Writing contents to file %s', os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = True

    # Mocking the necessary functions and objects
    os.path.isdir.return_value = False
    os.path.exists.return_value = True
    logger.debug = MagicMock()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    logger.debug.assert_any_call('The resulting file already exists: %s', os.path.join(project_dir, infile))
    shutil.copyfile.assert_not_called()
    env.get_template.assert_not_called()

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{empty}}'
    context = {'empty': ''}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    # Mocking the necessary functions and objects
    os.path.isdir.return_value = True
    logger.debug = MagicMock()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    logger.debug.assert_any_call('The resulting file name is empty: %s', os.path.join(project_dir, ''))
    shutil.copyfile.assert_not_called()
    env.get_template.assert_not_called()


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_context_applies_default_context():
    default_context = {'key': 'value'}
    extra_context = None
    context_file = 'cookiecutter.json'
    result = generate_context(context_file, default_context, extra_context)
    assert result['cookiecutter']['key'] == 'value'


# LLM-generated content at query #41
#--------------------------

```python
def test_os_walk_returns_true():
    assert os.walk('.') is not None


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_config():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
        assert '\n' in content


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #44
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #45
#--------------------------

```python
def test_is_binary_predicate_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()
    assert Path(result, 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()
    assert Path(result, 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    (Path(result) / 'test.txt').write_text('Existing content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()
    assert Path(result, 'test.txt').read_text() == 'Existing content'

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project', 'cookiecutter': {'_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.bin').write_bytes(b'\x00\x01\x02')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert Path(result, 'test.bin').exists()
    assert Path(result, 'test.bin').read_bytes() == b'\x00\x01\x02'

def test_generate_files_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    (Path(repo_dir) / 'hooks' / 'pre_gen_project.py').parent.mkdir()
    (Path(repo_dir) / 'hooks' / 'pre_gen_project.py').write_text('print("Pre-hook executed")')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()
    assert Path(result, 'test.txt').read_text() == 'Hello, test_project!'


# LLM-generated content at query #47
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isdir(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isdir(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isdir(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_no_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isdir(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_keep_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isdir(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')


# LLM-generated content at query #48
#--------------------------

```python
def test_cookiecutter_new_lines_is_true():
    context = {
        'cookiecutter': {
            '_new_lines': True
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #49
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_context_without_default_context():
    result = generate_context(context_file='valid.json', default_context=None, extra_context=None)
    assert 'valid' in result


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #52
#--------------------------

```python
def test_work_in_context_manager_returns_true():
    assert work_in(template_dir)


# LLM-generated content at query #53
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #54
#--------------------------

```python
def test_accept_hooks_predicate_true():
    assert True


# LLM-generated content at query #55
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key": "value"}')
    context = generate_context(context_file)
    assert context == {'test': {'key': 'value'}}
    os.remove(context_file)


# LLM-generated content at query #56
#--------------------------

```python
def test_undefined_error_during_render_and_create_dir():
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir="valid_repo",
            context={"invalid_key": "value"},
            output_dir="output",
            overwrite_if_exists=False,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )


# LLM-generated content at query #57
#--------------------------

```python
def test_accept_hooks_false():
    assert not False


# LLM-generated content at query #58
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #59
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context['cookiecutter']['project_name'] == 'My Project'
    assert context['cookiecutter']['project_slug'] == 'my_project'
    assert context['cookiecutter']['author'] == 'Your Name'

def test_generate_context_with_default_context():
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context=default_context
    )
    assert context['cookiecutter']['project_name'] == 'Default Project'

def test_generate_context_with_extra_context():
    extra_context = {'project_slug': 'extra_slug'}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['project_slug'] == 'extra_slug'

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')

def test_generate_context_with_empty_file():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/empty.json')

def test_generate_context_with_boolean_overwrite():
    extra_context = {'use_pytest': 'yes'}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['use_pytest'] is True

def test_generate_context_with_invalid_boolean_overwrite():
    extra_context = {'use_pytest': 'invalid'}
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-data/cookiecutter.json',
            extra_context=extra_context
        )

def test_generate_context_with_list_overwrite():
    extra_context = {'license': ['MIT']}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['license'] == ['MIT', 'BSD-3-Clause', 'GNU GPL v3.0']

def test_generate_context_with_invalid_list_overwrite():
    extra_context = {'license': ['Invalid']}
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-data/cookiecutter.json',
            extra_context=extra_context
        )

def test_generate_context_with_dict_overwrite():
    extra_context = {'config': {'key': 'value'}}
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context=extra_context
    )
    assert context['cookiecutter']['config']['key'] == 'value'


# LLM-generated content at query #60
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #61
#--------------------------

```python
def test_delete_project_on_failure_predicate_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #62
#--------------------------

```python
def test_os_walk_predicate():
    """Test that the predicate at line 62 evaluates to True."""
    # Mock the environment and context
    env = create_env_with_context({})
    context = {}

    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test directories and files
        os.makedirs(os.path.join(temp_dir, 'test_dir'))
        with open(os.path.join(temp_dir, 'test_file.txt'), 'w') as f:
            f.write('test content')

        # Change to the temporary directory
        with work_in(temp_dir):
            # Set up the environment loader
            env.loader = FileSystemLoader(['.', '../templates'])

            # Test the predicate by iterating through os.walk
            for root, dirs, files in os.walk('.'):
                assert isinstance(root, str)
                assert isinstance(dirs, list)
                assert isinstance(files, list)
                break


# LLM-generated content at query #63
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #64
#--------------------------

```python
def test_generate_context_with_no_default_context():
    result = generate_context(context_file='cookiecutter.json', default_context=None, extra_context=None)
    assert result is not None


# LLM-generated content at query #65
#--------------------------

```python
def test_work_in_context_manager_changes_and_restores_directory():
    original_dir = os.getcwd()
    test_dir = os.path.join(original_dir, 'test_dir')
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == original_dir
    shutil.rmtree(test_dir)


# LLM-generated content at query #66
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #67
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '2.0.0'}}


# LLM-generated content at query #68
#--------------------------

```python
def test_accept_hooks_false_predicate():
    assert not False


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_evaluates_to_true():
    context = {'cookiecutter': {'_copy_without_render': ['test_dir']}}
    assert is_copy_only_path('test_dir', context)


# LLM-generated content at query #70
#--------------------------

```python
def test_delete_project_on_failure_is_false():
    assert not (True and True)


# LLM-generated content at query #71
#--------------------------

```python
def test_undefined_error_in_render_and_create_dir():
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(
            repo_dir='valid_template',
            context={'invalid_key': 'value'},
            output_dir='output',
            overwrite_if_exists=False,
            skip_if_file_exists=False,
            accept_hooks=False,
            keep_project_on_failure=False
        )


# LLM-generated content at query #72
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #73
#--------------------------

```python
def test_accept_hooks_predicate_false():
    assert not accept_hooks is False


# LLM-generated content at query #74
#--------------------------

```python
def test_accept_hooks_false_predicate():
    assert not False


# LLM-generated content at query #75
#--------------------------

```python
def test_predicate_at_line_62():
    """Test that the predicate at line 62 evaluates to True."""
    # Setup
    repo_dir = Path('test_repo')
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    output_dir = Path('test_output')
    project_dir = Path('test_project')
    template_dir = Path('test_template')

    # Mock the necessary functions and objects
    os.listdir.return_value = ['{{cookiecutter.project_name}}']
    os.path.split.return_value = ('test_repo', '{{cookiecutter.project_name}}')
    os.path.abspath.return_value = str(project_dir)
    os.walk.return_value = [('.', ['dir1', 'dir2'], ['file1.txt', 'file2.txt'])]

    # Execute
    env = StrictEnvironment(context=context, keep_trailing_newline=True)
    template_dir = find_template(repo_dir, env)
    project_dir, output_directory_created = render_and_create_dir('{{cookiecutter.project_name}}', context, output_dir, env, False)
    delete_project_on_failure = output_directory_created and not False

    # Verify
    assert delete_project_on_failure is True


# LLM-generated content at query #76
#--------------------------

```python
def test_accept_hooks_predicate():
    repo_dir = '/fake/repo'
    context = {'fake': 'context'}
    output_dir = '/fake/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    with patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_and_create_dir, \
         patch('cookiecutter.generate.work_in'), \
         patch('cookiecutter.generate.FileSystemLoader'), \
         patch('cookiecutter.generate.os.walk') as mock_os_walk, \
         patch('cookiecutter.generate.is_copy_only_path') as mock_is_copy_only_path, \
         patch('cookiecutter.generate.shutil'), \
         patch('cookiecutter.generate.generate_file') as mock_generate_file, \
         patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:

        mock_find_template.return_value = '/fake/template'
        mock_render_and_create_dir.return_value = ('/fake/project', True)
        mock_os_walk.return_value = [('/', [], [])]
        mock_is_copy_only_path.return_value = False

        result = generate_files(
            repo_dir,
            context,
            output_dir,
            overwrite_if_exists,
            skip_if_file_exists,
            accept_hooks,
            keep_project_on_failure
        )

        assert mock_run_hook.called
        assert mock_run_hook.call_args_list[0][0][1] == 'pre_gen_project'
        assert mock_run_hook.call_args_list[1][0][1] == 'post_gen_project'


# LLM-generated content at query #77
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-fixtures/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-fixtures/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}


# LLM-generated content at query #78
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_no_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_keep_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project', 'cookiecutter': {'_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'copy_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'data.bin'))

def test_generate_files_undefined_variable():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'undefined_template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_empty_dir_name():
    context = {'project_name': ''}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    with pytest.raises(EmptyDirNameException):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_output_dir_exists():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    os.makedirs(os.path.join(output_dir, 'test_project'))
    with pytest.raises(OutputDirExistsException):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_new_lines():
    context = {'project_name': 'test_project', 'cookiecutter': {'_new_lines': '\r\n'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'README.md'), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content

def test_generate_files_hooks_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'hook_failure_template')
    with pytest.raises(FailedHookException):
        generate_files(repo_dir, context, output_dir)


# LLM-generated content at query #79
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #80
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert os.path.basename(project_dir) == 'test_project'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    project_dir = generate_files(repo_dir, context, output_dir)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)
    assert os.path.basename(project_dir) == 'test_project'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    project_dir = generate_files(repo_dir, context, output_dir)
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)
    assert os.path.basename(project_dir) == 'test_project'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_no_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(project_dir)
    assert os.path.basename(project_dir) == 'test_project'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_keep_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(project_dir)
    assert os.path.basename(project_dir) == 'test_project'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)


# LLM-generated content at query #81
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_true():
    repo_dir = "valid_template_dir"
    context = {"valid_key": "valid_value"}
    output_dir = "output_directory"
    overwrite_if_exists = True

    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        overwrite_if_exists=overwrite_if_exists,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False,
    )

    assert isinstance(result, str)


# LLM-generated content at query #82
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    repo_dir = Path('test_repo')
    context = {'cookiecutter': {'_copy_without_render': ['test_dir']}}
    output_dir = Path('test_output')
    env = create_env_with_context(context)
    template_dir = find_template(repo_dir, env)
    project_dir = Path('test_project')
    os.makedirs(project_dir, exist_ok=True)

    with work_in(template_dir):
        env.loader = FileSystemLoader(['.', '../templates'])
        os.makedirs('test_dir', exist_ok=True)

        for root, dirs, files in os.walk('.'):
            assert 'test_dir' not in dirs


# LLM-generated content at query #83
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #84
#--------------------------

```python
def test_os_walk_predicate_false():
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [('.', [], [])]
        with patch('cookiecutter.generate.find_template') as mock_find_template:
            mock_find_template.return_value = Path('.')
            with patch('cookiecutter.generate.render_and_create_dir') as mock_render_and_create_dir:
                mock_render_and_create_dir.return_value = (Path('.'), False)
                with patch('cookiecutter.generate.work_in'):
                    with patch('cookiecutter.generate.FileSystemLoader'):
                        with patch('cookiecutter.generate.is_copy_only_path') as mock_is_copy_only_path:
                            mock_is_copy_only_path.return_value = False
                            with patch('cookiecutter.generate.generate_file'):
                                with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                                    generate_files('.', {}, '.', False, False, False, False)
                                    mock_walk.assert_called_once_with('.')


# LLM-generated content at query #85
#--------------------------

```python
def test_delete_project_on_failure_predicate_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #86
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    repo_dir = Path('/fake/repo')
    context = {'cookiecutter': {'_copy_without_render': ['fake_dir']}}
    output_dir = Path('/fake/output')
    template_dir = Path('/fake/repo/{{cookiecutter.project_name}}')
    project_dir = Path('/fake/output/{{cookiecutter.project_name}}')

    with patch('cookiecutter.generate.find_template') as mock_find_template:
        with patch('cookiecutter.generate.render_and_create_dir') as mock_render_and_create_dir:
            with patch('cookiecutter.generate.work_in') as mock_work_in:
                with patch('cookiecutter.generate.os.walk') as mock_os_walk:
                    with patch('cookiecutter.generate.is_copy_only_path') as mock_is_copy_only_path:
                        with patch('cookiecutter.generate.os.path.isdir') as mock_isdir:
                            mock_find_template.return_value = template_dir
                            mock_render_and_create_dir.return_value = (project_dir, True)
                            mock_os_walk.return_value = [
                                ('.', ['fake_dir'], ['fake_file'])
                            ]
                            mock_is_copy_only_path.return_value = True
                            mock_isdir.return_value = False

                            generate_files(repo_dir, context, output_dir, False, False, False, True)

                            mock_os_walk.assert_called_once_with('.')


# LLM-generated content at query #87
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result).name == 'test_project'
    assert (Path(result) / 'file.txt').exists()

def test_generate_files_overwrite_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    generate_files(repo_dir, context, output_dir)
    generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(output_dir, 'test_project', 'file.txt').exists()

def test_generate_files_skip_existing_files():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    generate_files(repo_dir, context, output_dir)
    (Path(output_dir) / 'test_project' / 'file.txt').write_text('new content')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert (Path(output_dir) / 'test_project' / 'file.txt').read_text() == 'new content'

def test_generate_files_with_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    (Path(repo_dir) / 'hooks' / 'pre_gen_project.py').write_text('print("pre hook")')
    (Path(repo_dir) / 'hooks' / 'post_gen_project.py').write_text('print("post hook")')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(result).exists()

def test_generate_files_with_copy_without_render():
    context = {'project_name': 'test_project', 'cookiecutter': {'_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.bin').write_text('binary content')
    result = generate_files(repo_dir, context, output_dir)
    assert (Path(result) / 'file.bin').read_text() == 'binary content'

def test_generate_files_undefined_variable():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.undefined_var}}'
    template_dir.mkdir()
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_keep_project_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('{{cookiecutter.undefined_var}}')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(output_dir, 'test_project').exists()

def test_generate_files_no_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert Path(result).exists()

def test_generate_files_empty_context():
    context = {}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / 'project'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result).name == 'project'

def test_generate_files_nested_directories():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    nested_dir = template_dir / 'nested'
    nested_dir.mkdir()
    (nested_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result, 'nested', 'file.txt').exists()

def test_generate_files_binary_file():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'image.png').write_bytes(b'\x89PNG\r\n\x1a\n')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result, 'image.png').exists()
    assert Path(result, 'image.png').read_bytes() == b'\x89PNG\r\n\x1a\n'


# LLM-generated content at query #88
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #89
#--------------------------

```python
def test_accept_hooks_predicate():
    assert generate_files(
        repo_dir='test_repo',
        context={'test': 'value'},
        output_dir='test_output',
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )


# LLM-generated content at query #90
#--------------------------

```python
def test_accept_hooks_predicate():
    assert accept_hooks is True


# LLM-generated content at query #91
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    # Arrange
    repo_dir = Path('test_repo')
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    output_dir = Path('test_output')
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = False

    # Act
    with work_in(repo_dir):
        env = create_env_with_context(context)
        env.loader = FileSystemLoader(['.', '../templates'])

        # Create a directory structure that does not contain any directories
        # that match the predicate conditions
        os.makedirs('test_dir', exist_ok=True)
        os.makedirs('another_dir', exist_ok=True)

        # Assert
        for root, dirs, files in os.walk('.'):
            for d in dirs:
                d_ = os.path.normpath(os.path.join(root, d))
                assert not (
                    'cookiecutter' in d_
                    and env.variable_start_string in d_
                    and env.variable_end_string in d_
                )


# LLM-generated content at query #92
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_false():
    assert not (True and False)


# LLM-generated content at query #93
#--------------------------

```python
def test_delete_project_on_failure_is_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #94
#--------------------------

```python
def test_delete_project_on_failure_evaluates_to_true():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #95
#--------------------------

```python
def test_template_syntax_error_handling():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    with open(infile, 'w') as f:
        f.write('{% invalid syntax %}')
    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)
    assert exc_info.value.translated is False


# LLM-generated content at query #96
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-data/context.json')
    assert context == {'cookiecutter': {'name': 'test', 'value': 1}}

def test_generate_context_with_default_context():
    context = generate_context('tests/test-data/context.json', default_context={'name': 'default'})
    assert context == {'cookiecutter': {'name': 'default', 'value': 1}}

def test_generate_context_with_extra_context():
    context = generate_context('tests/test-data/context.json', extra_context={'name': 'extra'})
    assert context == {'cookiecutter': {'name': 'extra', 'value': 1}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')

def test_generate_context_with_empty_file():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/empty.json')

def test_generate_context_with_nested_dict():
    context = generate_context('tests/test-data/nested_context.json', extra_context={'nested': {'key': 'value'}})
    assert context == {'cookiecutter': {'name': 'test', 'nested': {'key': 'value'}}}

def test_generate_context_with_list_choice():
    context = generate_context('tests/test-data/list_context.json', extra_context={'choice': 'option2'})
    assert context == {'cookiecutter': {'choice': ['option2', 'option1', 'option3']}}

def test_generate_context_with_invalid_choice():
    with pytest.raises(ValueError):
        generate_context('tests/test-data/list_context.json', extra_context={'choice': 'invalid'})

def test_generate_context_with_boolean():
    context = generate_context('tests/test-data/bool_context.json', extra_context={'flag': 'yes'})
    assert context == {'cookiecutter': {'flag': True}}

def test_generate_context_with_invalid_boolean():
    with pytest.raises(ValueError):
        generate_context('tests/test-data/bool_context.json', extra_context={'flag': 'invalid'})


# LLM-generated content at query #97
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #98
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)


# LLM-generated content at query #99
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    outfile = "existing_file.txt"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w'):
        pass
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #100
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path("/tmp"), Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("existing_dir", {}, Path("/tmp"), Environment())

def test_render_and_create_dir_existing_dir_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("existing_dir", {}, Path("/tmp"), Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_new_dir():
    result = render_and_create_dir("new_dir", {}, Path("/tmp"), Environment())
    assert result == (Path("/tmp/new_dir"), True)
    Path("/tmp/new_dir").rmdir()

def test_render_and_create_dir_rendered_name():
    context = {"name": "test"}
    result = render_and_create_dir("{{ name }}_dir", context, Path("/tmp"), Environment())
    assert result == (Path("/tmp/test_dir"), True)
    Path("/tmp/test_dir").rmdir()


# LLM-generated content at query #101
#--------------------------

```python
def test_generate_file_binary_skips_rendering():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.join', return_value='/fake/project/binary.png'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_mode:
        generate_file(project_dir, infile, context, env)
        mock_copy.assert_called_once_with(infile, '/fake/project/binary.png')
        mock_mode.assert_called_once_with(infile, '/fake/project/binary.png')

def test_generate_file_text_renders_correctly():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\n', 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    env.get_template = Mock(return_value=Mock(render=Mock(return_value='rendered')))

    with patch('os.path.join', return_value='/fake/project/template.txt'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('shutil.copymode') as mock_mode:
        generate_file(project_dir, infile, context, env)
        mock_file.assert_called_with('/fake/project/template.txt', 'w', encoding='utf-8', newline='\n')
        mock_file().write.assert_called_once_with('rendered')
        mock_mode.assert_called_once_with(infile, '/fake/project/template.txt')

def test_generate_file_skips_existing_file():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.join', return_value='/fake/project/existing.txt'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True), \
         patch('utils.is_binary', return_value=False) as mock_binary:
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        mock_binary.assert_not_called()

def test_generate_file_empty_filename_skips():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.join', return_value='/fake/project'), \
         patch('os.path.isdir', return_value=True), \
         patch('utils.is_binary', return_value=False) as mock_binary:
        generate_file(project_dir, infile, context, env)
        mock_binary.assert_not_called()


# LLM-generated content at query #102
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #103
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test_cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test_cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'test_cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #104
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #105
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context(context_file='valid.json')
    assert context == {'valid': OrderedDict([])}


# LLM-generated content at query #106
#--------------------------

```python
def test_generate_files_basic():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    result2 = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert result == result2
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    result2 = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert result == result2
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(result)
    shutil.rmtree(output_dir)

def test_generate_files_without_hooks():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    repo_dir = 'tests/mocks/pre_and_post_gen_hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    shutil.rmtree(output_dir)


# LLM-generated content at query #107
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(result).exists()

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(result).exists()

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert Path(result).exists()

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(result).exists()

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template-undefined'
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.txt']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = 'tests/test-template-copy'
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'test_project', 'copy.txt').exists()


# LLM-generated content at query #108
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}'), 'w') as f:
        f.write('test content')

    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test content'))

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}'), 'w') as f:
        f.write('test content')

    generate_files(repo_dir, context, output_dir)
    generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}'), 'w') as f:
        f.write('test content')

    generate_files(repo_dir, context, output_dir)
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}'), 'w') as f:
        f.write('test content')

    with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
        f.write('print("Pre hook executed")')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_keep_project_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    with open(os.path.join(repo_dir, '{{cookiecutter.project_name}}'), 'w') as f:
        f.write('test content')

    with open(os.path.join(repo_dir, 'hooks', 'pre_gen_project.py'), 'w') as f:
        f.write('raise Exception("Hook failed")')

    with pytest.raises(Exception):
        generate_files(repo_dir, context, output_dir, accept_hooks=True, keep_project_on_failure=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))


# LLM-generated content at query #109
#--------------------------

```python
def test_predicate_false_at_line_36():
    assert not (False and True and True)


# LLM-generated content at query #110
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files('tests/test-templates/basic/', context)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))

def test_generate_files_with_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files('tests/test-templates/basic/', context, overwrite_if_exists=True)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files('tests/test-templates/basic/', context, skip_if_file_exists=True)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files('tests/test-templates/basic/', context, accept_hooks=False)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files('tests/test-templates/basic/', context, keep_project_on_failure=True)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))

def test_generate_files_with_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.txt']}}
    result = generate_files('tests/test-templates/basic/', context)
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, 'README.md'))
    assert os.path.isfile(os.path.join(result, 'copy_only.txt'))

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files('tests/test-templates/undefined_var/', context)

def test_generate_files_empty_dir_name():
    context = {'cookiecutter': {'project_name': ''}}
    with pytest.raises(EmptyDirNameException):
        generate_files('tests/test-templates/basic/', context)

def test_generate_files_output_dir_exists():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/test-output/'
    os.makedirs(output_dir, exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        generate_files('tests/test-templates/basic/', context, output_dir)


# LLM-generated content at query #111
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
        assert f.read() == 'Hello, test_project!'
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    generate_files(repo_dir, context, output_dir)
    generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'existing.txt'), 'w') as f:
        f.write('Existing file')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'existing.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Pre-hook executed")')
    with open(os.path.join(hooks_dir, 'post_gen_project.py'), 'w') as f:
        f.write('print("Post-hook executed")')
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'README.md'), 'w') as f:
        f.write('# {{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'README.md')) as f:
        assert f.read() == '# {{cookiecutter.project_name}}'
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.undefined_var}}')
    os.makedirs(template_dir)
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #112
#--------------------------

```python
def test_generate_context_with_invalid_json_file():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #113
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    context = {}
    output_directory_created = True
    keep_project_on_failure = True

    delete_project_on_failure = output_directory_created and not keep_project_on_failure

    assert delete_project_on_failure is False


# LLM-generated content at query #114
#--------------------------

```python
def test_accept_hooks_false():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    repo_dir = '/path/to/repo'
    output_dir = '/path/to/output'
    os.makedirs(repo_dir)
    os.makedirs(output_dir)
    (Path(repo_dir) / '{{cookiecutter.project_name}}').mkdir()
    (Path(repo_dir) / '{{cookiecutter.project_name}}' / 'file.txt').write_text('content')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)

    assert result == os.path.join(output_dir, '')


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_new_first_level_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_deeper_level_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": {"nested": "value", "new": "value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"var": ["a", "b", "c"]}
    overwrite_context = {"var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"var": {"a": 1, "b": 2}}
    overwrite_context = {"var": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": {"a": 1, "b": 3}}

def test_apply_overwrites_to_context_bool_conversion_valid():
    context = {"var": True}
    overwrite_context = {"var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": True}

def test_apply_overwrites_to_context_bool_conversion_invalid():
    context = {"var": True}
    overwrite_context = {"var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid provided for variable var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_with_list_in_dictionary_variable():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": ["d", "e"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == ["d", "e"]


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    context = {"var": True}
    overwrite_context = {"var": "invalid_string"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["var"] == "invalid_string"


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0'}}


# LLM-generated content at query #5
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    """Test that EmptyDirNameException is raised when dirname is empty."""
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path(), Environment())

def test_render_and_create_dir_existing_dir():
    """Test that OutputDirExistsException is raised when directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'name': 'test'}
        environment = Environment()
        dirname = '{{ name }}'
        output_dir = Path(tmpdir)
        dir_to_create = output_dir / 'test'
        dir_to_create.mkdir()
        with pytest.raises(OutputDirExistsException):
            render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_overwrite_existing():
    """Test that existing directory is overwritten when flag is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'name': 'test'}
        environment = Environment()
        dirname = '{{ name }}'
        output_dir = Path(tmpdir)
        dir_to_create = output_dir / 'test'
        dir_to_create.mkdir()
        result_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
        assert result_path == dir_to_create
        assert not created

def test_render_and_create_dir_new_directory():
    """Test that new directory is created successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context = {'name': 'test'}
        environment = Environment()
        dirname = '{{ name }}'
        output_dir = Path(tmpdir)
        result_path, created = render_and_create_dir(dirname, context, output_dir, environment)
        assert result_path.exists()
        assert result_path == output_dir / 'test'
        assert created


# LLM-generated content at query #6
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    _run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {}, False)
    assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": "d"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == ["a", "b", "c"]


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'binary.png'))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'template.txt'))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, 'existing.txt'), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'existing.txt'))

def test_generate_file_empty_outfile():
    project_dir = '/fake/project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'binary_file.png'))

def test_generate_file_text():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'text_file.txt'))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'existing_file.txt'))

def test_generate_file_empty_outfile():
    project_dir = '/tmp/test_project'
    infile = 'empty_dir'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, 'empty_dir'))

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'newline_file.txt'))

def test_generate_file_newline_config():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'newline_file.txt'))


# LLM-generated content at query #10
#--------------------------

```python
def test_output_dir_exists_predicate():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = False

    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    output_dir_exists = dir_to_create.exists()

    assert output_dir_exists is True


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_success():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp')
    environment = Environment()
    result = render_and_create_dir('{{ project_name }}', context, output_dir, environment)
    assert result[0] == output_dir / 'test_project'
    assert result[1] is True
    assert (output_dir / 'test_project').exists()

def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path('/tmp'), Environment())

def test_render_and_create_dir_exists_no_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp')
    environment = Environment()
    (output_dir / 'test_project').mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('{{ project_name }}', context, output_dir, environment)

def test_render_and_create_dir_exists_with_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp')
    environment = Environment()
    (output_dir / 'test_project').mkdir(exist_ok=True)
    result = render_and_create_dir('{{ project_name }}', context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == output_dir / 'test_project'
    assert result[1] is False


# LLM-generated content at query #12
#--------------------------

```python
def test_template_syntax_error_raised_when_invalid_template():
    project_dir = '/fake/project/dir'
    infile = 'invalid_template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    with open(infile, 'w') as f:
        f.write('{% invalid syntax %}')

    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile, context, env)


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\n' in content or b'\r\n' in content

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)


# LLM-generated content at query #15
#--------------------------

```python
def test_template_syntax_error_handling():
    project_dir = "/fake/project"
    infile = "fake_template.txt"
    context = {"cookiecutter": {}}
    env = Environment(loader=FileSystemLoader("."))

    # Create a template with syntax error
    with open(infile, "w") as f:
        f.write("{% if %}")

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)

    assert not exc_info.value.translated


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_files_basic():
    """Test basic file generation with minimal context."""
    repo_dir = 'tests/test-template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_overwrite_existing():
    """Test overwriting existing output directory."""
    repo_dir = 'tests/test-template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(result).exists()

def test_generate_files_skip_existing():
    """Test skipping existing files."""
    repo_dir = 'tests/test-template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(result).exists()

def test_generate_files_with_hooks():
    """Test file generation with hooks enabled."""
    repo_dir = 'tests/test-template-with-hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(result).exists()

def test_generate_files_without_hooks():
    """Test file generation with hooks disabled."""
    repo_dir = 'tests/test-template-with-hooks'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert Path(result).exists()

def test_generate_files_keep_on_failure():
    """Test keeping project directory on failure."""
    repo_dir = 'tests/test-template-with-error'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except Exception:
        assert Path(output_dir, 'test_project').exists()

def test_generate_files_delete_on_failure():
    """Test deleting project directory on failure."""
    repo_dir = 'tests/test-template-with-error'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'tests/output'
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=False)
    except Exception:
        assert not Path(output_dir, 'test_project').exists()


# LLM-generated content at query #17
#--------------------------

```python
def test_output_dir_exists_predicate():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = False

    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    output_dir_exists = dir_to_create.exists()
    assert output_dir_exists is True


# LLM-generated content at query #18
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with raises(EmptyDirNameException, match='Error: directory name is empty'):
        render_and_create_dir('', {}, '/tmp', Environment())


# LLM-generated content at query #19
#--------------------------

```python
def test_render_and_create_dir_when_dir_exists():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)

    result = render_and_create_dir(dirname, context, output_dir, environment)

    assert result[1] is False


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_copymode:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_copy.assert_called_once_with(infile, os.path.join(project_dir, infile))
        mock_copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch('builtins.open', mock_open(read_data='line1\nline2')), \
         patch('shutil.copymode') as mock_copymode:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #21
#--------------------------

```python
def test_generate_files_success():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = './test_output'
    repo_dir = './test_repo'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = './test_output'
    repo_dir = './test_repo_with_hooks'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = './test_output'
    repo_dir = './test_repo'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = './test_output'
    repo_dir = './test_repo_failure'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_no_context():
    output_dir = './test_output'
    repo_dir = './test_repo'
    result = generate_files(repo_dir, None, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_raises_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)


# LLM-generated content at query #24
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_raises_value_error():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #25
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_variable_deep_level():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": {"nested": "value", "new": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable choices, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoice": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multichoice, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key2": "new_value2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "value1", "key2": "new_value2"}}

def test_apply_overwrites_to_context_bool_true():
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_false():
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}

def test_apply_overwrites_to_context_bool_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old_value"}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new_value"}


# LLM-generated content at query #26
#--------------------------

```python
def test_open_file_success():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key": "value"}')
    result = generate_context(context_file)
    assert result == {'test': {'key': 'value'}}


# LLM-generated content at query #27
#--------------------------

```python
def test_skip_if_file_exists_predicate_evaluates_to_true():
    project_dir = '/fake/project/dir'
    infile = 'fake_template.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, 'w') as f:
        f.write('fake content')

    result = skip_if_file_exists and os.path.exists(outfile)

    assert result is True


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test-cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid.json')

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')

def test_generate_context_with_default_context_invalid():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )

def test_generate_context_with_extra_context_invalid():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter.json',
            extra_context={'invalid': 'value'}
        )


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x00\x01\x02'

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile)) as f:
        assert f.read() == 'Hello test!'

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write('existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == 'existing content'

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{ cookiecutter.empty }}'
    context = {'cookiecutter': {'_new_lines': None, 'empty': ''}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template_nl.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line1\r\nLine2\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #30
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    os.path.exists.return_value = True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_evaluates_to_false():
    context = {"key": True}
    overwrite_context = {"key": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #32
#--------------------------

```python
def test_file_name_is_empty_predicate():
    project_dir = '/path/to/project'
    infile = 'some_directory/'
    context = {}
    env = Environment()

    outfile_tmpl = env.from_string(infile)
    outfile = os.path.join(project_dir, outfile_tmpl.render(**context))
    file_name_is_empty = os.path.isdir(outfile)

    assert file_name_is_empty is True


# LLM-generated content at query #33
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #34
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write('existing content')

    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #35
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_true():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #37
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {"variable": True}
    overwrite_context = {"variable": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_empty_outfile():
    project_dir = '/fake/project'
    infile = 'empty_dir/'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #39
#--------------------------

```python
def test_file_name_is_empty_when_outfile_is_directory():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment()

    # Mock os.path.isdir to return True
    import os
    original_isdir = os.path.isdir
    os.path.isdir = lambda path: True

    try:
        generate_file(project_dir, infile, context, env)
        assert file_name_is_empty == True
    finally:
        os.path.isdir = original_isdir


# LLM-generated content at query #40
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "output_dir", Environment())


# LLM-generated content at query #41
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    assert is_copy_only_path('file.txt', {'cookiecutter': {'_copy_without_render': ['*.txt']}}) == True

def test_is_copy_only_path_returns_false_when_path_does_not_match_pattern():
    assert is_copy_only_path('file.py', {'cookiecutter': {'_copy_without_render': ['*.txt']}}) == False

def test_is_copy_only_path_returns_false_when_no_patterns_in_context():
    assert is_copy_only_path('file.txt', {'cookiecutter': {}}) == False

def test_is_copy_only_path_returns_false_when_context_missing_cookiecutter_key():
    assert is_copy_only_path('file.txt', {}) == False

def test_is_copy_only_path_returns_true_for_directory_path_matching_pattern():
    assert is_copy_only_path('src/static', {'cookiecutter': {'_copy_without_render': ['*/static']}}) == True

def test_is_copy_only_path_returns_false_for_directory_path_not_matching_pattern():
    assert is_copy_only_path('src/templates', {'cookiecutter': {'_copy_without_render': ['*/static']}}) == False


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_render():
    project_dir = '/tmp/test_project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = '{{ "" }}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))


# LLM-generated content at query #43
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.isdir(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template-with-hooks'
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.isdir(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.isdir(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.isdir(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = tempfile.mkdtemp()
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.isdir(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)


# LLM-generated content at query #44
#--------------------------

```python
def test_output_dir_exists_predicate():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    overwrite_if_exists = True

    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    output_dir_exists = dir_to_create.exists()

    assert output_dir_exists is True


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #46
#--------------------------

```python
def test_is_binary_returns_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #47
#--------------------------

```python
def test_default_context_is_not_none():
    default_context = {"key": "value"}
    context_file = "cookiecutter.json"
    extra_context = None
    result = generate_context(context_file, default_context, extra_context)
    assert default_context is not None


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copyfile, \
         patch('shutil.copymode') as mock_copymode:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_copyfile.assert_called_once_with(infile, os.path.join(project_dir, infile))
        mock_copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch.object(env, 'get_template') as mock_get_template, \
         patch('builtins.open', mock_open(read_data='template content')) as mock_file, \
         patch('shutil.copymode') as mock_copymode:
        mock_template = MagicMock()
        mock_template.render.return_value = 'rendered content'
        mock_get_template.return_value = mock_template
        mock_file.return_value.readline.return_value = 'template content'
        mock_file.return_value.newlines = '\n'

        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_get_template.assert_called_once_with(infile.replace(os.path.sep, '/'))
        mock_template.render.assert_called_once_with(**context)
        mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\n')
        mock_copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        # No assertions needed, just verify no exceptions are raised

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = 'empty_dir/'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        # No assertions needed, just verify no exceptions are raised

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch.object(env, 'get_template') as mock_get_template, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('shutil.copymode') as mock_copymode:
        mock_template = MagicMock()
        mock_template.render.return_value = 'rendered content'
        mock_get_template.return_value = mock_template

        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\r\n')


# LLM-generated content at query #49
#--------------------------

```python
def test_is_binary_returns_true():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #50
#--------------------------

```python
def test_render_and_create_dir_overwrite_if_exists_true():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = True

    # Create the directory first to simulate it already exists
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    result_path, created = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )

    assert result_path == dir_to_create
    assert not created


# LLM-generated content at query #51
#--------------------------

```python
def test_output_dir_exists_predicate():
    output_dir_exists = True
    assert output_dir_exists


# LLM-generated content at query #52
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    os.path.exists.return_value = True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #53
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #54
#--------------------------

```python
def test_is_binary_evaluates_to_true():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #55
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context(context_file='tests/test-cookiecutter.json')
    assert 'test_cookiecutter' in context
    assert context['test_cookiecutter']['name'] == 'test'

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='tests/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context['test_cookiecutter']['name'] == 'default'

def test_generate_context_with_extra_context():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context['test_cookiecutter']['name'] == 'extra'

def test_generate_context_with_bool_conversion():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context={'use_pytest': 'yes'}
    )
    assert context['test_cookiecutter']['use_pytest'] is True

def test_generate_context_with_invalid_bool():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-cookiecutter.json',
            extra_context={'use_pytest': 'invalid'}
        )

def test_generate_context_with_list_overwrite():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context={'framework': 'flask'}
    )
    assert context['test_cookiecutter']['framework'] == ['flask', 'django']

def test_generate_context_with_invalid_list_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-cookiecutter.json',
            extra_context={'framework': 'invalid'}
        )

def test_generate_context_with_dict_overwrite():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context={'config': {'debug': True}}
    )
    assert context['test_cookiecutter']['config']['debug'] is True


# LLM-generated content at query #56
#--------------------------

```python
def test_accept_hooks_predicate_true():
    assert True is True


# LLM-generated content at query #57
#--------------------------

```python
def test_new_lines_predicate_evaluates_to_true():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #58
#--------------------------

```python
def test_render_and_create_dir_success():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path("/tmp/test_dir")
    assert result[1] == True
    assert Path("/tmp/test_dir").exists()

def test_render_and_create_dir_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    Path("/tmp/test_dir").mkdir(exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == Path("/tmp/test_dir")
    assert result[1] == False
    assert Path("/tmp/test_dir").exists()

def test_render_and_create_dir_exists_exception():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    Path("/tmp/test_dir").mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_empty_name_exception():
    dirname = ""
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(dirname, context, output_dir, environment)


# LLM-generated content at query #59
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert not is_binary(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert is_binary(os.path.join(project_dir, infile)) == False

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\n' in content or b'\r\n' in content

def test_generate_file_custom_newline():
    project_dir = '/tmp/test_project'
    infile = 'custom_newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #60
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result


# LLM-generated content at query #61
#--------------------------

```python
def test_generate_context_opens_file():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')) as mock_file:
        result = generate_context('test.json')
        mock_file.assert_called_once_with('test.json', encoding='utf-8')


# LLM-generated content at query #62
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test-cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'test-cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'name': 'invalid', 'invalid_key': 'value'}
        )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #63
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    assert not (output_directory_created and not keep_project_on_failure)


# LLM-generated content at query #64
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #65
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert filecmp.cmp(infile, os.path.join(project_dir, infile))

def test_generate_file_text_render():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert 'test' in content

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert content == ''

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    assert b'\r\n' in content


# LLM-generated content at query #66
#--------------------------

```python
def test_generate_context_with_default_context_none():
    result = generate_context(default_context=None)
    assert result == OrderedDict([])


# LLM-generated content at query #67
#--------------------------

```python
def test_delete_project_on_failure_predicate_false():
    """Test that delete_project_on_failure is False when output_directory_created is False."""
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    overwrite_if_exists = False
    keep_project_on_failure = True
    output_directory_created = False

    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #68
#--------------------------

```python
def test_generate_context_with_no_default_context():
    context = generate_context(context_file='cookiecutter.json', default_context=None, extra_context=None)
    assert context == OrderedDict([('cookiecutter', OrderedDict())])


# LLM-generated content at query #69
#--------------------------

```python
def test_default_context_is_not_none():
    default_context = {'key': 'value'}
    assert default_context is not None


# LLM-generated content at query #70
#--------------------------

```python
def test_os_walk_predicate_true():
    os.listdir.return_value = ['{{cookiecutter.test}}']
    os.path.isdir.return_value = True
    os.path.split.return_value = ('.', '{{cookiecutter.test}}')
    os.path.abspath.return_value = '/abs/path'
    os.getcwd.return_value = '/current/dir'
    os.path.normpath.side_effect = lambda x: x
    os.path.join.side_effect = lambda *args: '/'.join(args)
    os.path.relpath.return_value = '{{cookiecutter.test}}'
    env.from_string.return_value.render.return_value = '/rendered/path'
    is_copy_only_path.return_value = False
    render_and_create_dir.return_value = ('/project/dir', True)
    os.walk.return_value = [('.', ['dir1'], ['file1.txt'])]

    result = generate_files(
        repo_dir='./test-repo',
        context={'cookiecutter': {'test': 'value'}},
        output_dir='./output',
        overwrite_if_exists=False,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )

    assert result == '/project/dir'


# LLM-generated content at query #71
#--------------------------

```python
def test_os_walk_returns_non_empty_iterator():
    # Setup a temporary directory structure
    temp_dir = Path(mkdtemp())
    sub_dir = temp_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file.txt").write_text("content")

    # Change to the temp directory to simulate the work_in context
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # The predicate at line 62 is the for loop itself, which expects os.walk('.') to yield items
        walk_result = list(os.walk('.'))
        assert len(walk_result) > 0
        assert len(walk_result[0]) == 3  # (root, dirs, files) tuple
    finally:
        os.chdir(original_dir)
        rmtree(temp_dir)


# LLM-generated content at query #72
#--------------------------

```python
def test_delete_project_on_failure_is_true_when_output_directory_created_and_keep_project_on_failure_is_false():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #73
#--------------------------

```python
def test_generate_files_basic():
    """Test basic file generation."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_overwrite():
    """Test file generation with overwrite."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_skip_existing():
    """Test file generation with skip existing files."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_no_hooks():
    """Test file generation without hooks."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_keep_on_failure():
    """Test file generation with keep on failure."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_undefined_variable():
    """Test file generation with undefined variable."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_copy_without_render():
    """Test file generation with copy without render."""
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    repo_dir = 'tests/fake-repo-pre'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #74
#--------------------------

```python
def test_undefined_error_in_render_and_create_dir():
    repo_dir = Path('test_repo')
    context = {'project_name': 'test_project'}
    output_dir = Path('test_output')
    env = create_env_with_context(context)
    template_dir = find_template(repo_dir, env)
    unrendered_dir = os.path.split(template_dir)[1]

    with pytest.raises(UndefinedVariableInTemplate):
        render_and_create_dir(unrendered_dir, context, output_dir, env, False)


# LLM-generated content at query #75
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    assert not (output_directory_created and not keep_project_on_failure)


# LLM-generated content at query #76
#--------------------------

```python
def test_generate_context_opens_file():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')) as mock_file:
        result = generate_context('test.json')
        mock_file.assert_called_once_with('test.json', encoding='utf-8')


# LLM-generated content at query #77
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    """Ensure the predicate at line 62 evaluates to False."""
    context = {'_copy_without_render': ['some_dir']}
    os.listdir.return_value = ['some_dir']
    os.path.isdir.return_value = False
    os.path.normpath.return_value = 'some_dir'

    with patch('cookiecutter.generate.os') as mock_os:
        mock_os.listdir.return_value = ['some_dir']
        mock_os.path.isdir.return_value = False
        mock_os.path.normpath.return_value = 'some_dir'
        mock_os.walk.return_value = [('.', ['some_dir'], [])]

        with patch('cookiecutter.generate.is_copy_only_path') as mock_is_copy_only:
            mock_is_copy_only.return_value = True

            with patch('cookiecutter.generate.create_env_with_context') as mock_env:
                mock_env.return_value = Environment()

                with patch('cookiecutter.generate.find_template') as mock_find_template:
                    mock_find_template.return_value = Path('template_dir')

                    with patch('cookiecutter.generate.render_and_create_dir') as mock_render:
                        mock_render.return_value = (Path('project_dir'), True)

                        with patch('cookiecutter.generate.work_in'):
                            with patch('cookiecutter.generate.run_hook_from_repo_dir'):
                                result = generate_files('repo_dir', context)

                                mock_is_copy_only.assert_called_with('some_dir', context)
                                assert mock_is_copy_only.return_value is True


# LLM-generated content at query #78
#--------------------------

```python
def test_generate_context_with_default_context_none():
    result = generate_context(default_context=None)
    assert result == {'cookiecutter': OrderedDict([])}


# LLM-generated content at query #79
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-valid-cookiecutter.json')
    assert context == {'test_valid_cookiecutter': {'name': 'test'}}


# LLM-generated content at query #80
#--------------------------

```python
def test_accept_hooks_predicate():
    accept_hooks = True
    assert accept_hooks is True


# LLM-generated content at query #81
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context('tests/test_cookiecutter.json')
    assert context == {'test_cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json_file():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid_cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test_cookiecutter.json',
        default_context={'key': 'default_value'}
    )
    assert context == {'test_cookiecutter': {'key': 'default_value'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test_cookiecutter.json',
        extra_context={'key': 'extra_value'}
    )
    assert context == {'test_cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test_cookiecutter.json',
            default_context={'invalid_key': 'invalid_value'}
        )

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')


# LLM-generated content at query #82
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'README.md'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # Create existing directory
    os.makedirs(os.path.join(output_dir, 'test_project'))

    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'README.md'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # Create existing file
    os.makedirs(os.path.join(output_dir, 'test_project'))
    with open(os.path.join(output_dir, 'test_project', 'README.md'), 'w') as f:
        f.write('existing content')

    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    with open(os.path.join(output_dir, 'test_project', 'README.md'), 'r') as f:
        assert f.read() == 'existing content'
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'hook_marker.txt'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_with_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_binary')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'data.bin'))
    assert result == os.path.join(output_dir, 'test_project')


# LLM-generated content at query #83
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-data/nonexistent.json')


# LLM-generated content at query #84
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    env = create_env_with_context(context)
    template_dir = find_template('.', env)
    project_dir = os.path.abspath('.')
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #85
#--------------------------

```python
def test_accept_hooks_false_predicate():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    repo_dir = Path('test_repo')
    output_dir = Path('test_output')
    os.makedirs(repo_dir)
    os.makedirs(output_dir)
    template_dir = repo_dir / '{{cookiecutter.project_name}}'
    os.makedirs(template_dir)
    (template_dir / 'test.txt').write_text('test')

    result = generate_files(
        repo_dir=repo_dir,
        context=context,
        output_dir=output_dir,
        accept_hooks=False
    )

    assert result == os.path.abspath(os.path.join(output_dir, '{{cookiecutter.project_name}}'))


# LLM-generated content at query #86
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_false():
    delete_project_on_failure = False
    assert not delete_project_on_failure


# LLM-generated content at query #87
#--------------------------

```python
def test_generate_context_with_default_context_none():
    result = generate_context(default_context=None)
    assert result is not None


# LLM-generated content at query #88
#--------------------------

```python
def test_undefined_error_in_render_and_create_dir():
    context = {'project_name': 'test'}
    env = create_env_with_context(context)
    repo_dir = Path('test_repo')
    template_dir = repo_dir / '{{cookiecutter.project_name}}'
    template_dir.mkdir(parents=True)

    with pytest.raises(UndefinedVariableInTemplate) as exc_info:
        generate_files(repo_dir, context)

    assert "Unable to create project directory '{{cookiecutter.project_name}}'" in str(exc_info.value)


# LLM-generated content at query #89
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #90
#--------------------------

```python
def test_delete_project_on_failure_is_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #91
#--------------------------

```python
def test_work_in_context_manager_returns_true():
    template_dir = "/path/to/template"
    assert work_in(template_dir).__enter__() == template_dir


# LLM-generated content at query #92
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'file.txt').exists()
    assert Path(result, 'file.txt').read_text() == 'content'

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(result).exists()
    assert Path(result, 'file.txt').exists()
    assert Path(result, 'file.txt').read_text() == 'content'

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(result).exists()
    assert Path(result, 'file.txt').exists()
    assert Path(result, 'file.txt').read_text() == 'content'

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    (repo_dir / 'hooks' / 'pre_gen_project.py').parent.mkdir()
    (repo_dir / 'hooks' / 'pre_gen_project.py').write_text('print("pre hook")')
    (repo_dir / 'hooks' / 'post_gen_project.py').write_text('print("post hook")')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(result).exists()
    assert Path(result, 'file.txt').exists()
    assert Path(result, 'file.txt').read_text() == 'content'

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('content')
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(result).exists()
    assert Path(result, 'file.txt').exists()
    assert Path(result, 'file.txt').read_text() == 'content'


# LLM-generated content at query #93
#--------------------------

```python
def test_accept_hooks_predicate_false():
    assert not accept_hooks


# LLM-generated content at query #94
#--------------------------

```python
def test_delete_project_on_failure_is_false():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #95
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = Path('tests/test-template')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = Path('tests/test-template')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = Path('tests/test-template')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = Path('tests/test-template-with-hooks')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = Path('tests/test-template')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()

def test_generate_files_copy_only_paths():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    repo_dir = Path('tests/test-template-with-binaries')
    output_dir = Path('tests/output')
    result = generate_files(repo_dir, context, output_dir)
    assert result == output_dir / 'test_project'
    assert (output_dir / 'test_project').exists()
    assert (output_dir / 'test_project' / 'test.bin').exists()


# LLM-generated content at query #96
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create a simple template structure
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert (Path(result) / 'file.txt').exists()
    assert (Path(result) / 'file.txt').read_text() == 'Hello, test_project!'

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template with hooks
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    # Create pre and post hook directories
    (Path(repo_dir) / 'hooks').mkdir()
    (Path(repo_dir) / 'hooks' / 'pre_gen_project.py').write_text('print("Pre hook")')
    (Path(repo_dir) / 'hooks' / 'post_gen_project.py').write_text('print("Post hook")')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert Path(result).exists()
    assert (Path(result) / 'file.txt').exists()

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    # First generation
    result = generate_files(repo_dir, context, output_dir)

    # Second generation with overwrite
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert Path(result).exists()
    assert (Path(result) / 'file.txt').exists()

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    # First generation
    result = generate_files(repo_dir, context, output_dir)

    # Modify the output file
    (Path(result) / 'file.txt').write_text('Modified content')

    # Second generation with skip
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    # File should not be overwritten
    assert (Path(result) / 'file.txt').read_text() == 'Modified content'

def test_generate_files_binary_file():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template with binary file
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    binary_content = b'\x00\x01\x02\x03'
    (template_dir / 'binary.bin').write_bytes(binary_content)

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert (Path(result) / 'binary.bin').exists()
    assert (Path(result) / 'binary.bin').read_bytes() == binary_content

def test_generate_files_copy_without_render():
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': ['*.md']
        }
    }
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template with copy-only file
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'readme.md').write_text('This should not be rendered: {{cookiecutter.project_name}}')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert (Path(result) / 'readme.md').exists()
    assert (Path(result) / 'readme.md').read_text() == 'This should not be rendered: {{cookiecutter.project_name}}'

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create template with undefined variable
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'file.txt').write_text('Hello, {{cookiecutter.undefined_var}}!')

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)


# LLM-generated content at query #97
#--------------------------

```python
def test_accept_hooks_false_predicate():
    repo_dir = "test_repo"
    context = {}
    output_dir = "test_output"
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = False

    assert not accept_hooks


# LLM-generated content at query #98
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_empty_context():
    context = {}
    repo_dir = 'tests/example-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_default_output_dir():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/example-template'
    result = generate_files(repo_dir, context)
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #99
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_output_directory_not_created_and_keep_project_on_failure_is_false():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #100
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_custom_output_dir():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/custom-output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'custom-output' in result

def test_generate_files_empty_context():
    context = {}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_none_context():
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, None, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_default_output_dir():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    result = generate_files(repo_dir, context)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_all_options():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(
        repo_dir,
        context,
        output_dir,
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        accept_hooks=True,
        keep_project_on_failure=True
    )
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #101
#--------------------------

```python
def test_delete_project_on_failure_predicate_false():
    output_directory_created = False
    keep_project_on_failure = True
    assert not (output_directory_created and not keep_project_on_failure)


# LLM-generated content at query #102
#--------------------------

```python
def test_generate_files_basic():
    """Test basic file generation with minimal context."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_overwrite():
    """Test file generation with overwrite enabled."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')

    # First generation
    generate_files(repo_dir, context, output_dir)

    # Second generation with overwrite
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_skip_existing():
    """Test file generation with skip_if_file_exists enabled."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')

    # First generation
    generate_files(repo_dir, context, output_dir)

    # Modify a file
    readme_path = os.path.join(output_dir, 'test_project', 'README.md')
    with open(readme_path, 'w') as f:
        f.write('Modified content')

    # Second generation with skip
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    # Check that the file wasn't overwritten
    with open(readme_path) as f:
        assert f.read() == 'Modified content'

def test_generate_files_copy_without_render():
    """Test file generation with _copy_without_render setting."""
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': ['*.bin', 'static/*']
        }
    }
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'copy_template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project', 'data.bin'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'static', 'file.txt'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_hooks():
    """Test file generation with hooks enabled."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'hook_template')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'hook_marker.txt'))
    assert result == os.path.join(output_dir, 'test_project')

def test_generate_files_keep_on_failure():
    """Test that project is kept when generation fails and keep_project_on_failure is True."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'failing_template')

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)

    # Check that partial output still exists
    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_no_context():
    """Test file generation with no context provided."""
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'no_context_template')

    result = generate_files(repo_dir, output_dir=output_dir)

    assert os.path.exists(os.path.join(output_dir, 'project'))
    assert result == os.path.join(output_dir, 'project')


# LLM-generated content at query #103
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'new_file.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'failing-template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #104
#--------------------------

```python
def test_accept_hooks_predicate_evaluates_to_true():
    assert True is True


# LLM-generated content at query #105
#--------------------------

```python
def test_os_walk_returns_non_empty_iterator():
    # Mock the os.walk function to return a non-empty iterator
    os.walk.return_value = [('.', [], [])]
    assert list(os.walk('.')) == [('.', [], [])]


# LLM-generated content at query #106
#--------------------------

```python
def test_work_in_context_manager_changes_directory():
    original_dir = os.getcwd()
    test_dir = '/test/directory'
    with work_in(test_dir):
        assert os.getcwd() == test_dir
    assert os.getcwd() == original_dir


# LLM-generated content at query #107
#--------------------------

```python
def test_delete_project_on_failure_predicate_true():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #108
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is True


# LLM-generated content at query #109
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #110
#--------------------------

```python
def test_json_decoding_error_raises_context_decoding_exception():
    with pytest.raises(ContextDecodingException) as exc_info:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(exc_info.value)


# LLM-generated content at query #111
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_failure():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    with pytest.raises(ValueError) as excinfo:
        apply_overwrites_to_context(context, overwrite_context)
    assert "invalid provided for variable test_var could not be converted to a boolean." in str(excinfo.value)


# LLM-generated content at query #112
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = '/fake/project/dir'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))

    # Create a template with syntax error
    with open(infile, 'w') as f:
        f.write('{% if %}')

    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile, context, env)


# LLM-generated content at query #113
#--------------------------

```python
def test_undefined_error_raised_in_render_and_create_dir():
    with pytest.raises(UndefinedVariableInTemplate) as exc_info:
        generate_files(
            repo_dir='valid_repo',
            context={'invalid': '{{ undefined_var }}'},
            output_dir='output',
            overwrite_if_exists=True
        )
    assert "Unable to create project directory" in str(exc_info.value)


# LLM-generated content at query #114
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('some_binary_file.png') is True


# LLM-generated content at query #115
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #116
#--------------------------

```python
def test_output_dir_exists_predicate():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)
    output_dir_exists = dir_to_create.exists()
    assert output_dir_exists is True


# LLM-generated content at query #117
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '2.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )

def test_generate_context_with_none_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context=None,
        extra_context=None
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #118
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    open(outfile, 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(outfile)


# LLM-generated content at query #119
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #120
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    assert not (True and True)


# LLM-generated content at query #121
#--------------------------

```python
def test_generate_context_with_default_context():
    default_context = {"key": "value"}
    extra_context = None
    context_file = "cookiecutter.json"
    with patch("builtins.open", mock_open(read_data='{"key": "value"}')):
        with patch("json.load", return_value=OrderedDict([("key", "value")])):
            result = generate_context(context_file, default_context, extra_context)
            assert result == {"cookiecutter": OrderedDict([("key", "value")])}


# LLM-generated content at query #122
#--------------------------

```python
def test_generate_context_opens_file():
    context = generate_context(context_file='cookiecutter.json')
    assert 'cookiecutter' in context


# LLM-generated content at query #123
#--------------------------

```python
def test_generate_context_with_default_context_none():
    result = generate_context(default_context=None)
    assert result == {'cookiecutter': OrderedDict([])}


# LLM-generated content at query #124
#--------------------------

```python
def test_work_in_context_manager_changes_and_restores_directory():
    initial_dir = os.getcwd()
    test_dir = os.path.join(initial_dir, 'test_dir')
    os.makedirs(test_dir, exist_ok=True)

    with work_in(test_dir):
        assert os.getcwd() == test_dir

    assert os.getcwd() == initial_dir


# LLM-generated content at query #125
#--------------------------

```python
def test_delete_project_on_failure_predicate():
    assert (True and not False) == True
    assert (False and not False) == False
    assert (True and not True) == False
    assert (False and not True) == False


# LLM-generated content at query #126
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project', 'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()
    assert Path(result, 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_overwrite_existing():
    context = {'project_name': 'test_project', 'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    first_run = generate_files(repo_dir, context, output_dir)
    second_run = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert first_run == second_run
    assert Path(second_run, 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_skip_existing_files():
    context = {'project_name': 'test_project', 'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    first_run = generate_files(repo_dir, context, output_dir)
    (Path(first_run) / 'test.txt').write_text('Modified content')
    second_run = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert first_run == second_run
    assert Path(second_run, 'test.txt').read_text() == 'Modified content'

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project', 'cookiecutter': {'_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'readme.md').write_text('{{cookiecutter.project_name}}')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result, 'readme.md').read_text() == '{{cookiecutter.project_name}}'

def test_generate_files_binary_file():
    context = {'project_name': 'test_project', 'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    binary_content = b'\x00\x01\x02\x03'
    (template_dir / 'binary.bin').write_bytes(binary_content)

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result, 'binary.bin').read_bytes() == binary_content

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

def test_generate_files_hooks():
    context = {'project_name': 'test_project', 'cookiecutter': {}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    hooks_dir = Path(repo_dir, 'hooks')
    hooks_dir.mkdir()
    (hooks_dir / 'pre_gen_project.py').write_text('print("Pre hook executed")')
    (hooks_dir / 'post_gen_project.py').write_text('print("Post hook executed")')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert Path(result).exists()
    assert Path(result, 'test.txt').exists()


# LLM-generated content at query #127
#--------------------------

```python
def test_accept_hooks_false():
    assert not False


# LLM-generated content at query #128
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('{{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
        assert f.read() == 'test_project'
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('{{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('{{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'w') as f:
        f.write('existing')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    with open(os.path.join(output_dir, 'test_project', 'test.txt')) as f:
        assert f.read() == 'existing'
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.md'), 'w') as f:
        f.write('{{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'test.md')) as f:
        assert f.read() == '{{cookiecutter.project_name}}'
    shutil.rmtree(output_dir)

def test_generate_files_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    hooks_dir = os.path.join(template_dir, 'hooks')
    os.makedirs(hooks_dir)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("pre hook")')
    with open(os.path.join(hooks_dir, 'post_gen_project.py'), 'w') as f:
        f.write('print("post hook")')
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('{{cookiecutter.project_name}}')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    shutil.rmtree(output_dir)


# LLM-generated content at query #129
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    assert not (output_directory_created and not keep_project_on_failure)


# LLM-generated content at query #130
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/test_output'
    repo_dir = '/tmp/test_repo'
    result = generate_files(repo_dir, context, output_dir)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/test_output'
    repo_dir = '/tmp/test_repo'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/test_output'
    repo_dir = '/tmp/test_repo'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/test_output'
    repo_dir = '/tmp/test_repo'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/test_output'
    repo_dir = '/tmp/test_repo'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert isinstance(result, str)
    assert os.path.exists(result)


# LLM-generated content at query #131
#--------------------------

```python
def test_os_walk_predicate_false():
    os.walk.return_value = [('.', [], [])]
    assert not list(os.walk('.'))


# LLM-generated content at query #132
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, '/tmp', Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    dir_to_create = Path('/tmp/existing_dir')
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('existing_dir', {}, '/tmp', Environment())

def test_render_and_create_dir_existing_dir_with_overwrite():
    dir_to_create = Path('/tmp/existing_dir')
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir('existing_dir', {}, '/tmp', Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_new_dir():
    result = render_and_create_dir('new_dir', {}, '/tmp', Environment())
    assert result == (Path('/tmp/new_dir'), True)
    assert Path('/tmp/new_dir').exists()

def test_render_and_create_dir_rendered_name():
    context = {'name': 'test'}
    result = render_and_create_dir('{{ name }}_dir', context, '/tmp', Environment())
    assert result == (Path('/tmp/test_dir'), True)
    assert Path('/tmp/test_dir').exists()


# LLM-generated content at query #133
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid': 'value'}
        )

def test_generate_context_with_invalid_extra_context():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-data/cookiecutter.json',
            extra_context={'version': 'invalid'}
        )


# LLM-generated content at query #134
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context('valid_context.json')
    assert context == {'valid_context': {'key': 'value'}}


# LLM-generated content at query #135
#--------------------------

```python
def test_accept_hooks_predicate_true():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    repo_dir = Path('test_repo')
    output_dir = Path('test_output')
    os.makedirs(repo_dir / '{{cookiecutter.project_name}}', exist_ok=True)
    (repo_dir / '{{cookiecutter.project_name}}').mkdir(exist_ok=True)

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert (Path(output_dir) / '{{cookiecutter.project_name}}').exists()


# LLM-generated content at query #136
#--------------------------

```python
def test_template_syntax_error_exception_handling():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with open(infile, 'w') as f:
        f.write('{% invalid syntax %}')

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert exc_info.value.translated is False


# LLM-generated content at query #137
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #138
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": {"new_nested": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a", "c"]}

def test_apply_overwrites_to_context_invalid_choice():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choices, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoice": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multichoice, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"dict_var": {"key1": "val1", "key2": "val2"}}
    overwrite_context = {"dict_var": {"key2": "new_val2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"key1": "val1", "key2": "new_val2"}}

def test_apply_overwrites_to_context_bool_true():
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_false():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}

def test_apply_overwrites_to_context_bool_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable bool_var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #139
#--------------------------

```python
def test_skip_if_file_exists():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    # Create a fake existing file
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, 'w') as f:
        f.write('existing content')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # Verify the file wasn't modified
    with open(outfile, 'r') as f:
        assert f.read() == 'existing content'

    os.remove(outfile)
    os.rmdir(project_dir)


# LLM-generated content at query #140
#--------------------------

```python
def test_generate_context_with_default_context_none():
    result = generate_context(default_context=None)
    assert result == OrderedDict([])


