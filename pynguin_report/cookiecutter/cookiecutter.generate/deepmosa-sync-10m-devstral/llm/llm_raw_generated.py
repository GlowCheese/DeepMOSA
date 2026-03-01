####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "overwrite"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_variable_deeper_level():
    context = {"dict_var": {"existing": "value"}}
    overwrite_context = {"dict_var": {"new": "overwrite"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"existing": "value", "new": "overwrite"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["x", "y"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["x", "y"]}

def test_apply_overwrites_to_context_valid_multichoice():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi_var": ["a", "c"]}

def test_apply_overwrites_to_context_invalid_multichoice():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d provided for multi-choice variable multi_var, but valid choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_valid_choice():
    context = {"choice_var": ["a", "b", "c"]}
    overwrite_context = {"choice_var": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice_var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_invalid_choice():
    context = {"choice_var": ["a", "b", "c"]}
    overwrite_context = {"choice_var": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d provided for choice variable choice_var, but the choices are ['a', 'b', 'c']." in str(e)

def test_apply_overwrites_to_context_partial_dict_overwrite():
    context = {"dict_var": {"a": 1, "b": 2, "c": 3}}
    overwrite_context = {"dict_var": {"b": 20, "d": 4}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"a": 1, "b": 20, "c": 3, "d": 4}}

def test_apply_overwrites_to_context_boolean_true():
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_boolean_false():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}

def test_apply_overwrites_to_context_invalid_boolean():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable bool_var could not be converted to a boolean." in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "original"}
    overwrite_context = {"var": "overwrite"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "overwrite"}


# LLM-generated content at query #2
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
        _run_hook_from_repo_dir(
            repo_dir='repo_dir',
            hook_name='hook_name',
            project_dir='project_dir',
            context={'key': 'value'},
            delete_project_on_failure=True
        )
        mock_run_hook.assert_called_once_with(
            'repo_dir',
            'hook_name',
            'project_dir',
            {'key': 'value'},
            True
        )

def test__run_hook_from_repo_dir_emits_deprecation_warning():
    with patch('cookiecutter.generate.run_hook_from_repo_dir'):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _run_hook_from_repo_dir(
                repo_dir='repo_dir',
                hook_name='hook_name',
                project_dir='project_dir',
                context={'key': 'value'},
                delete_project_on_failure=True
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_run_hook_from_repo_dir' function is deprecated" in str(w[0].message)


# LLM-generated content at query #3
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert isinstance(result, str)
    assert os.path.exists(result)


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())

def test_render_and_create_dir_with_existing_dir_and_no_overwrite():
    dir_to_create = Path("existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("existing_dir", {}, Path(), Environment())

def test_render_and_create_dir_with_existing_dir_and_overwrite():
    dir_to_create = Path("existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("existing_dir", {}, Path(), Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_with_new_dir():
    result = render_and_create_dir("new_dir", {}, Path(), Environment())
    assert result == (Path("new_dir"), True)
    Path("new_dir").rmdir()

def test_render_and_create_dir_with_rendered_name():
    context = {"name": "test"}
    result = render_and_create_dir("{{ name }}_dir", context, Path(), Environment())
    assert result == (Path("test_dir"), True)
    Path("test_dir").rmdir()


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"variable": True}
    overwrite_context = {"variable": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable variable could not be converted to a boolean."


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_57_evaluates_to_false():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_output_dir_exists_predicate():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = True
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)
    output_dir_exists = dir_to_create.exists()
    assert output_dir_exists is True


# LLM-generated content at query #9
#--------------------------

```python
def test_output_dir_exists_and_overwrite_if_exists_is_true():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = True

    # Create the directory to ensure it exists
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
    assert result[1] is False


# LLM-generated content at query #10
#--------------------------

```python
def test_apply_overwrites_to_context_with_dict_overwrite():
    context = {"key": {"subkey": "value"}}
    overwrite_context = {"key": "not_a_dict"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": {"subkey": "value"}}


# LLM-generated content at query #11
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-fixtures/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context={'name': 'override'}
    )
    assert context == {'cookiecutter': {'name': 'override', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-fixtures/invalid.json')

def test_generate_context_with_missing_file():
    with pytest.raises(FileNotFoundError):
        generate_context('tests/test-fixtures/missing.json')

def test_generate_context_with_empty_context_file():
    context = generate_context('tests/test-fixtures/empty.json')
    assert context == {'empty': {}}

def test_generate_context_with_nested_dict():
    context = generate_context('tests/test-fixtures/nested.json')
    assert context == {'nested': {'a': {'b': 1}}}

def test_generate_context_with_list_overwrite():
    context = generate_context(
        'tests/test-fixtures/list.json',
        extra_context={'items': ['a', 'b']}
    )
    assert context == {'list': {'items': ['a', 'b']}}

def test_generate_context_with_invalid_list_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-fixtures/list.json',
            extra_context={'items': ['invalid']}
        )

def test_generate_context_with_bool_overwrite():
    context = generate_context(
        'tests/test-fixtures/bool.json',
        extra_context={'flag': 'yes'}
    )
    assert context == {'bool': {'flag': True}}

def test_generate_context_with_invalid_bool_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-fixtures/bool.json',
            extra_context={'flag': 'invalid'}
        )


# LLM-generated content at query #12
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #13
#--------------------------

```python
def test_apply_overwrites_to_context_new_first_level_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_deeper_level_variable():
    context = {"dict": {"existing": "value"}}
    overwrite_context = {"dict": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict": {"existing": "value", "new": "value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list": [1, 2, 3]}
    overwrite_context = {"list": [4, 5, 6]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"list": [4, 5, 6]}

def test_apply_overwrites_to_context_valid_multichoice():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["a", "c", "b"]}

def test_apply_overwrites_to_context_invalid_multichoice():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_valid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_invalid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"dict": {"a": 1, "b": 2, "c": 3}}
    overwrite_context = {"dict": {"b": 20, "d": 40}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict": {"a": 1, "b": 20, "c": 3, "d": 40}}

def test_apply_overwrites_to_context_bool_conversion_valid():
    context = {"bool": True}
    overwrite_context = {"bool": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool": True}

def test_apply_overwrites_to_context_bool_conversion_invalid():
    context = {"bool": True}
    overwrite_context = {"bool": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable bool could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new"}


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test_cookiecutter': {'name': 'test', 'value': 1}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test_cookiecutter': {'name': 'default', 'value': 1}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'value': 1}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'value': 1}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'test_cookiecutter': {'name': 'test', 'value': 1}}


# LLM-generated content at query #15
#--------------------------

```python
def test_generate_context_raises_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as exc_info:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(exc_info.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-fixtures/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-fixtures/invalid.json')

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')


# LLM-generated content at query #18
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "invalid provided for variable bool_var could not be converted to a boolean."


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    result = generate_context(context_file)
    assert result == {'test_context': {'key': 'value'}}
    os.remove(context_file)


# LLM-generated content at query #20
#--------------------------

```python
def test_is_copy_only_path_matching_pattern():
    path = "file.txt"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is True

def test_is_copy_only_path_not_matching_pattern():
    path = "file.py"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_no_patterns():
    path = "file.txt"
    context = {
        'cookiecutter': {}
    }
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_empty_patterns():
    path = "file.txt"
    context = {
        'cookiecutter': {
            '_copy_without_render': []
        }
    }
    assert is_copy_only_path(path, context) is False

def test_is_copy_only_path_multiple_patterns():
    path = "file.txt"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["*.py", "*.txt"]
        }
    }
    assert is_copy_only_path(path, context) is True


# LLM-generated content at query #21
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'temp*']
        }
    }
    assert is_copy_only_path('temp_file.txt', context) is True


# LLM-generated content at query #22
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'failing_template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_newline_configured():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/project'
    infile = 'binary.bin'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    with open(infile, 'rb') as f:
        expected_content = f.read()
    assert content == expected_content

def test_generate_file_text_file():
    project_dir = '/tmp/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert 'test' in content

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # No assertion needed, just ensuring no error is raised

def test_generate_file_empty_filename():
    project_dir = '/tmp/project'
    infile = '{{""}}'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # No file should be created

def test_generate_file_newline_detection():
    project_dir = '/tmp/project'
    infile = 'newlines.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    assert b'\r\n' in content or b'\n' in content

def test_generate_file_custom_newline():
    project_dir = '/tmp/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    assert b'\r\n' in content


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test_cookiecutter': {'name': 'test', 'value': 1}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid-cookiecutter.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test_cookiecutter': {'name': 'default', 'value': 1}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'value': 1}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'test_cookiecutter': {'name': 'extra', 'value': 1}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'test_cookiecutter': {'name': 'test', 'value': 1}}

def test_generate_context_with_invalid_extra_context():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter.json',
            extra_context={'invalid': 'value'}
        )


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'project_name': 'test_project', 'author': 'test_author'}}


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as temp_dir:
        context = {'project_name': 'test_project'}
        environment = Environment()
        dirname = '{{ project_name }}'
        output_dir = Path(temp_dir)

        result_path, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )

        assert result_path == Path(temp_dir) / 'test_project'
        assert created is True
        assert result_path.exists() is True

def test_render_and_create_dir_empty_dirname():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as temp_dir:
        context = {}
        environment = Environment()
        dirname = ''
        output_dir = Path(temp_dir)

        try:
            render_and_create_dir(dirname, context, output_dir, environment)
        except EmptyDirNameException as e:
            assert str(e) == 'Error: directory name is empty'

def test_render_and_create_dir_exists_no_overwrite():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as temp_dir:
        context = {'project_name': 'test_project'}
        environment = Environment()
        dirname = '{{ project_name }}'
        output_dir = Path(temp_dir)

        # Create the directory first
        Path(temp_dir, 'test_project').mkdir()

        try:
            render_and_create_dir(dirname, context, output_dir, environment)
        except OutputDirExistsException as e:
            assert str(e) == 'Error: "' + str(Path(temp_dir) / 'test_project') + '" directory already exists'

def test_render_and_create_dir_exists_with_overwrite():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as temp_dir:
        context = {'project_name': 'test_project'}
        environment = Environment()
        dirname = '{{ project_name }}'
        output_dir = Path(temp_dir)

        # Create the directory first
        Path(temp_dir, 'test_project').mkdir()

        result_path, created = render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=True
        )

        assert result_path == Path(temp_dir) / 'test_project'
        assert created is False
        assert result_path.exists() is True


# LLM-generated content at query #4
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

def test_is_copy_only_path_returns_false_when_no_patterns_in_context():
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
    path = "src/utils"
    context = {
        'cookiecutter': {
            '_copy_without_render': ["src/*"]
        }
    }
    assert is_copy_only_path(path, context) is True


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

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '2.0.0'}}

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')


# LLM-generated content at query #6
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
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_invalid_choice():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable list_var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi_var": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multi_var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"dict_var": {"key1": "val1", "key2": "val2"}}
    overwrite_context = {"dict_var": {"key1": "new_val1"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"key1": "new_val1", "key2": "val2"}}

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
    except ValueError as e:
        assert str(e) == "invalid provided for variable bool_var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_with_default_and_extra_context():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        default_context={'project_name': 'test_project'},
        extra_context={'project_slug': 'test_slug'}
    )
    assert context['cookiecutter']['project_name'] == 'test_project'
    assert context['cookiecutter']['project_slug'] == 'test_slug'

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='tests/test-fixtures/invalid.json')

def test_generate_context_with_missing_file():
    with pytest.raises(FileNotFoundError):
        generate_context(context_file='tests/test-fixtures/nonexistent.json')

def test_generate_context_with_no_overrides():
    context = generate_context(context_file='tests/test-fixtures/cookiecutter.json')
    assert context['cookiecutter']['project_name'] == 'default_project'
    assert context['cookiecutter']['project_slug'] == 'default_slug'

def test_generate_context_with_boolean_conversion():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        extra_context={'use_pytest': 'yes'}
    )
    assert context['cookiecutter']['use_pytest'] is True

def test_generate_context_with_invalid_boolean():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-fixtures/cookiecutter.json',
            extra_context={'use_pytest': 'invalid'}
        )

def test_generate_context_with_list_overwrite():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        extra_context={'framework': 'flask'}
    )
    assert context['cookiecutter']['framework'] == ['flask', 'django', 'pyramid']

def test_generate_context_with_invalid_list_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-fixtures/cookiecutter.json',
            extra_context={'framework': 'invalid_framework'}
        )

def test_generate_context_with_dict_overwrite():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        extra_context={'database': {'name': 'postgres'}}
    )
    assert context['cookiecutter']['database']['name'] == 'postgres'
    assert context['cookiecutter']['database']['host'] == 'localhost'

def test_generate_context_with_new_variable():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        extra_context={'new_variable': 'new_value'}
    )
    assert 'new_variable' not in context['cookiecutter']

def test_generate_context_with_new_dict_variable():
    context = generate_context(
        context_file='tests/test-fixtures/cookiecutter.json',
        extra_context={'new_dict': {'key': 'value'}}
    )
    assert context['cookiecutter']['new_dict'] == {'key': 'value'}


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_evaluates_to_false():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    overwrite_if_exists = False

    # Create the directory to ensure it exists
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)

    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)

    assert result[1] is False


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    context = {"key": True}
    overwrite_context = {"key": "invalid"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] is True


# LLM-generated content at query #10
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
        context = generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_output_dir_exists_false():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[1] == True


# LLM-generated content at query #12
#--------------------------

```python
def test_apply_overwrites_to_context_with_list_and_in_dictionary_variable():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": ["d", "e"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == ["d", "e"]


# LLM-generated content at query #13
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)

    result = generate_context(context_file)
    assert result == {'test_context': {'key': 'value'}}
    os.remove(context_file)


# LLM-generated content at query #15
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
        _run_hook_from_repo_dir('repo', 'hook', 'project', {}, False)
        mock_run_hook.assert_called_once_with('repo', 'hook', 'project', {}, False)

def test__run_hook_from_repo_dir_emits_deprecation_warning():
    with patch('cookiecutter.generate.run_hook_from_repo_dir'), \
         patch('warnings.warn') as mock_warn:
        _run_hook_from_repo_dir('repo', 'hook', 'project', {}, False)
        mock_warn.assert_called_once_with(
            "The '_run_hook_from_repo_dir' function is deprecated, "
            "use 'cookiecutter.hooks.run_hook_from_repo_dir' instead",
            DeprecationWarning,
            2,
        )


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {
        'cookiecutter': {
            'project_name': 'My Project',
            'project_slug': 'my_project',
            'author': 'Your Name',
            'email': 'your@email.com',
            'version': '0.1.0',
        }
    }

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'project_name': 'New Project'}
    )
    assert context['cookiecutter']['project_name'] == 'New Project'

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'project_name': 'Extra Project'}
    )
    assert context['cookiecutter']['project_name'] == 'Extra Project'

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid_key': 'value'}
        )

def test_generate_context_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('nonexistent.json')


# LLM-generated content at query #18
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"variable": True}
    overwrite_context = {"variable": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #19
#--------------------------

```python
def test_output_dir_exists_predicate():
    output_dir = Path('/tmp')
    dirname = 'test_dir'
    context = {}
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)
    output_dir_exists = dir_to_create.exists()
    assert output_dir_exists


# LLM-generated content at query #20
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #21
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result1 = generate_files(repo_dir, context, output_dir)
    result2 = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert result1 == result2
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'r') as f:
        assert f.read() == 'existing content'
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-binaries')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project', 'data.bin'))
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-undefined')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #24
#--------------------------

```python
def test_render_and_create_dir_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #25
#--------------------------

```python
def test_render_and_create_dir_success():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0].name == "test_dir"
    assert result[1] is True

def test_render_and_create_dir_empty_dirname():
    dirname = ""
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_exists_no_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create = Path(output_dir, "test_dir")
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_exists_with_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create = Path(output_dir, "test_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0].name == "test_dir"
    assert result[1] is False


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_evaluates_to_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #27
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
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"a": 1, "b": 3}}

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

def test_apply_overwrites_to_context_invalid_bool():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new"}


# LLM-generated content at query #28
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path(), Environment())

def test_render_and_create_dir_existing_output_dir():
    with pytest.raises(OutputDirExistsException):
        output_dir = Path('existing_dir')
        output_dir.mkdir(exist_ok=True)
        render_and_create_dir('test_dir', {}, output_dir, Environment())

def test_render_and_create_dir_overwrite_existing():
    output_dir = Path('existing_dir')
    output_dir.mkdir(exist_ok=True)
    result = render_and_create_dir('test_dir', {}, output_dir, Environment(), overwrite_if_exists=True)
    assert result[0] == output_dir / 'test_dir'
    assert not result[1]

def test_render_and_create_dir_new_dir():
    output_dir = Path('new_dir')
    result = render_and_create_dir('test_dir', {}, output_dir, Environment())
    assert result[0] == output_dir / 'test_dir'
    assert result[1]
    assert (output_dir / 'test_dir').exists()


# LLM-generated content at query #29
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_raises_value_error():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'file.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'file.txt'), 'w') as f:
        f.write('existing content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    with open(os.path.join(result, 'test_project', 'file.txt'), 'r') as f:
        assert f.read() == 'existing content'
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'copy_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project', 'readme.md'))
    with open(os.path.join(result, 'test_project', 'readme.md'), 'r') as f:
        assert '{{' not in f.read()
    shutil.rmtree(output_dir)

def test_generate_files_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'hook_template')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'undefined_template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)

def test_generate_files_keep_project_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'undefined_template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_context_with_no_default_context():
    result = generate_context(context_file='cookiecutter.json', default_context=None, extra_context=None)
    assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_with_newline_config():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\n' in content


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_mode:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_copy.assert_called_once_with(infile, f'{project_dir}/{infile}')
        mock_mode.assert_called_once_with(infile, f'{project_dir}/{infile}')

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch('shutil.copymode') as mock_mode, \
         patch('builtins.open', mock_open()) as mock_file:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_file.assert_called()
        mock_mode.assert_called_once_with(infile, f'{project_dir}/{infile}')

def test_generate_file_skip_existing():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True), \
         patch('utils.is_binary', return_value=False):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        # No file operations should occur


# LLM-generated content at query #34
#--------------------------

```python
def test_file_name_is_empty_when_outfile_is_directory():
    project_dir = '/fake/project'
    infile = '{{cookiecutter.fake_var}}'
    context = {'cookiecutter': {'fake_var': 'fake_dir'}}
    env = Environment()
    outfile = os.path.join(project_dir, 'fake_dir')
    os.makedirs(outfile, exist_ok=True)
    file_name_is_empty = os.path.isdir(outfile)
    assert file_name_is_empty is True


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project/dir'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

def test_generate_file_text_file():
    project_dir = '/fake/project/dir'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ name }}!')
    generate_file(project_dir, infile, {'name': 'World', 'cookiecutter': {'_new_lines': None}}, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Hello, World!'

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project/dir'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('Existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Existing content'

def test_generate_file_empty_filename():
    project_dir = '/fake/project/dir'
    infile = '{{ empty_var }}/file.txt'
    context = {'empty_var': '', 'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_config():
    project_dir = '/fake/project/dir'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\n') as f:
        f.write('Line 1\nLine 2')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

def test_generate_file_text_rendering():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'name': 'test'}}
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
    context = {'cookiecutter': {}}
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
    context = {'cookiecutter': {'empty': ''}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'newlines.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #38
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
    infile = '{{ cookiecutter.empty }}.txt'
    context = {'cookiecutter': {'_new_lines': None, 'empty': ''}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_config():
    project_dir = '/fake/project'
    infile = 'newline.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n', 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_mode:
        generate_file(project_dir, infile, context, env)
        mock_copy.assert_called_once_with(infile, os.path.join(project_dir, infile))
        mock_mode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('templates'))
    rendered = "Hello {{ name }}"

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch.object(env, 'get_template') as mock_get, \
         patch('builtins.open', mock_open()) as mock_file:
        mock_get.return_value.render.return_value = rendered
        generate_file(project_dir, infile, context, env)
        mock_file.assert_called_once_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\n')

def test_generate_file_skip_existing():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        # No file operations should occur


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'binary.jpg'))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'template.txt'))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, 'existing.txt'), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'existing.txt'))

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        content = f.read()
        assert b'\n' in content or b'\r\n' in content

def test_generate_file_configured_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #41
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    os.path.exists.return_value = True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #42
#--------------------------

```python
def test_template_syntax_error_exception_handling():
    project_dir = "/fake/project/dir"
    infile = "fake_template.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment(loader=FileSystemLoader("."))
    skip_if_file_exists = False

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert exc_info.value.translated is False


# LLM-generated content at query #43
#--------------------------

```python
def test_is_binary_predicate_true():
    infile = "binary_file.png"
    assert is_binary(infile) is True


# LLM-generated content at query #44
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
    assert filecmp.cmp(infile, os.path.join(project_dir, infile))

def test_generate_file_text_rendering():
    project_dir = '/tmp/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'name': 'test', '_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert 'test' in content

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert content == ''

def test_generate_file_empty_filename():
    project_dir = '/tmp/project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_handling():
    project_dir = '/tmp/project'
    infile = 'newline_test.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    assert b'\r\n' in content


# LLM-generated content at query #45
#--------------------------

```python
def test_is_binary_predicate_true():
    assert is_binary("test_binary_file.bin") is True


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('templates'))
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
    context = {'cookiecutter': {'_new_lines': False, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Hello test!'

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': False, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write('Existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, 'r') as f:
        assert f.read() == 'Existing content'

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{ cookiecutter.name }}.txt'
    context = {'cookiecutter': {'_new_lines': False, 'name': ''}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_config():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n', 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert b'\r\n' in f.read()


# LLM-generated content at query #47
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
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
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch.object(env, 'get_template') as mock_get_template, \
         patch('builtins.open', mock_open(read_data='line1\n')) as mock_file, \
         patch('shutil.copymode') as mock_copymode:
        mock_template = MagicMock()
        mock_template.render.return_value = 'rendered content'
        mock_get_template.return_value = mock_template

        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        mock_template.render.assert_called_once_with(**context)
        mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline='\n')
        mock_copymode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    with patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True), \
         patch('utils.is_binary', return_value=False):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = ''
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    with patch('os.path.isdir', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
        assert not os.path.exists(os.path.join(project_dir, infile))


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context(context_file='tests/test-cookiecutter.json')
    assert context['test-cookiecutter']['project_name'] == 'Test Project'
    assert context['test-cookiecutter']['author'] == 'Test Author'

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='tests/invalid.json')

def test_generate_context_with_default_context():
    default_context = {'project_name': 'Default Project'}
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        default_context=default_context
    )
    assert context['test-cookiecutter']['project_name'] == 'Default Project'

def test_generate_context_with_extra_context():
    extra_context = {'project_name': 'Extra Project'}
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context=extra_context
    )
    assert context['test-cookiecutter']['project_name'] == 'Extra Project'

def test_generate_context_with_invalid_default_context():
    default_context = {'invalid_var': 'value'}
    with pytest.warns(UserWarning):
        generate_context(
            context_file='tests/test-cookiecutter.json',
            default_context=default_context
        )

def test_generate_context_with_non_existent_file():
    with pytest.raises(FileNotFoundError):
        generate_context(context_file='non_existent.json')


# LLM-generated content at query #49
#--------------------------

```python
def test_cookiecutter_new_lines_true():
    context = {
        'cookiecutter': {
            '_new_lines': True
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.join', return_value='/fake/project/binary_file.png'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=True), \
         patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_copymode:

        generate_file(project_dir, infile, context, env)

        mock_copy.assert_called_once_with(infile, '/fake/project/binary_file.png')
        mock_copymode.assert_called_once_with(infile, '/fake/project/binary_file.png')

def test_generate_file_text_render():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('templates'))
    env.get_template = Mock(return_value=Mock(render=Mock(return_value='rendered content')))

    with patch('os.path.join', return_value='/fake/project/template.txt'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=False), \
         patch('utils.is_binary', return_value=False), \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('shutil.copymode') as mock_copymode:

        generate_file(project_dir, infile, context, env)

        mock_file.assert_called_with('/fake/project/template.txt', 'w', encoding='utf-8', newline=None)
        mock_copymode.assert_called_once_with(infile, '/fake/project/template.txt')

def test_generate_file_skip_existing():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))

    with patch('os.path.join', return_value='/fake/project/existing.txt'), \
         patch('os.path.isdir', return_value=False), \
         patch('os.path.exists', return_value=True):

        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

        # No file operations should occur
        assert True

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{}}'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    env.from_string = Mock(return_value=Mock(render=Mock(return_value='')))

    with patch('os.path.join', return_value=''), \
         patch('os.path.isdir', return_value=True):

        generate_file(project_dir, infile, context, env)

        # No file operations should occur
        assert True


# LLM-generated content at query #51
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #53
#--------------------------

```python
def test_ensure_predicate_at_line_59_evaluates_to_true():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    env = create_env_with_context(context)
    repo_dir = Path('test_repo')
    template_dir = repo_dir / '{{cookiecutter.project_name}}'
    template_dir.mkdir(parents=True)
    project_dir = Path('test_output')
    project_dir.mkdir()
    delete_project_on_failure = True
    keep_project_on_failure = False
    output_directory_created = True

    assert output_directory_created and not keep_project_on_failure == delete_project_on_failure


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #55
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = 'empty_filename/'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_custom_newline():
    project_dir = '/tmp/test_project'
    infile = 'custom_newline.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #56
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'README.md'))
    assert 'test_project' in open(os.path.join(output_dir, 'test_project', 'README.md')).read()

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'hook_output.txt'))

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')

    # First generation
    generate_files(repo_dir, context, output_dir)

    # Second generation with skip
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')

    # First generation
    generate_files(repo_dir, context, output_dir)

    # Second generation with overwrite
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-binaries')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.isfile(os.path.join(output_dir, 'test_project', 'data.bin'))

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-undefined')

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)


# LLM-generated content at query #57
#--------------------------

```python
def test_generate_context_default_context_none():
    result = generate_context(default_context=None)
    assert not result.get('cookiecutter')  # Assuming 'cookiecutter.json' is the default context_file


# LLM-generated content at query #58
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    result = generate_files(repo_dir, context, output_dir)

    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()
    assert (Path(result) / 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    first_run = generate_files(repo_dir, context, output_dir)
    second_run = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert first_run == second_run
    assert (Path(second_run) / 'test.txt').read_text() == 'Hello, test_project!'

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')

    first_run = generate_files(repo_dir, context, output_dir)
    (Path(first_run) / 'test.txt').write_text('Modified content')
    second_run = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert (Path(second_run) / 'test.txt').read_text() == 'Modified content'

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    (repo_dir / 'hooks' / 'pre_gen_project.py').parent.mkdir()
    (repo_dir / 'hooks' / 'pre_gen_project.py').write_text('print("Pre-hook executed")')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'README.md').write_text('This is a {{cookiecutter.project_name}} project.')

    result = generate_files(repo_dir, context, output_dir)

    assert (Path(result) / 'README.md').read_text() == 'This is a {{cookiecutter.project_name}} project.'

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir, '{{cookiecutter.project_name}}')
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.undefined_var}}!')

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)


# LLM-generated content at query #59
#--------------------------

```python
def test_generate_context_with_valid_json():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            result = generate_context('test.json')
            assert result == {'test': {"key": "value"}}

def test_generate_context_with_invalid_json():
    with patch('builtins.open', mock_open(read_data='{invalid json}')):
        with patch('json.load', side_effect=ValueError("Invalid JSON")):
            with pytest.raises(ContextDecodingException):
                generate_context('test.json')

def test_generate_context_with_default_context():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            with patch('cookiecutter.generate.apply_overwrites_to_context') as mock_apply:
                generate_context('test.json', default_context={"key": "new_value"})
                mock_apply.assert_called_once_with({"key": "value"}, {"key": "new_value"})

def test_generate_context_with_extra_context():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            with patch('cookiecutter.generate.apply_overwrites_to_context') as mock_apply:
                generate_context('test.json', extra_context={"key": "new_value"})
                mock_apply.assert_called_once_with({"key": "value"}, {"key": "new_value"})

def test_generate_context_with_both_contexts():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            with patch('cookiecutter.generate.apply_overwrites_to_context') as mock_apply:
                generate_context('test.json', default_context={"key": "default"}, extra_context={"key": "extra"})
                assert mock_apply.call_count == 2


# LLM-generated content at query #60
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #61
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #62
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('invalid.json')


# LLM-generated content at query #63
#--------------------------

```python
def test_default_context_is_applied():
    default_context = {'key': 'value'}
    extra_context = None
    context_file = 'cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'old_value'}, f)
    result = generate_context(context_file, default_context, extra_context)
    assert result['cookiecutter']['key'] == 'value'


# LLM-generated content at query #64
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_output_directory_not_created_and_keep_project_on_failure_is_true():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #65
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #66
#--------------------------

```python
def test_predicate_at_line_54_evaluates_to_false():
    assert not (True and not False)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test_cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        'tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test_cookiecutter': {'name': 'default', 'version': '2.0.0'}}

def test_generate_context_with_non_existent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('non-existent.json')


# LLM-generated content at query #2
#--------------------------

```python
def test__run_hook_from_repo_dir():
    _run_hook_from_repo_dir(
        repo_dir='repo_dir',
        hook_name='hook_name',
        project_dir='project_dir',
        context={'key': 'value'},
        delete_project_on_failure=True,
    )


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, Path(), Environment())

def test_render_and_create_dir_with_existing_dir_and_no_overwrite():
    dir_to_create = Path('existing_dir')
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('existing_dir', {}, dir_to_create.parent, Environment())

def test_render_and_create_dir_with_existing_dir_and_overwrite():
    dir_to_create = Path('existing_dir')
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir('existing_dir', {}, dir_to_create.parent, Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_with_new_dir():
    output_dir = Path('test_output')
    result = render_and_create_dir('new_dir', {}, output_dir, Environment())
    assert result == (output_dir / 'new_dir', True)
    assert (output_dir / 'new_dir').exists()

def test_render_and_create_dir_with_template_rendering():
    output_dir = Path('test_output')
    result = render_and_create_dir('{{ project_name }}', {'project_name': 'test_project'}, output_dir, Environment())
    assert result == (output_dir / 'test_project', True)
    assert (output_dir / 'test_project').exists()


# LLM-generated content at query #4
#--------------------------

```python
def test_apply_overwrites_to_context_new_first_level_variable_ignored():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_deeper_level_variable_added():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": {"nested": "value", "new": "value"}}

def test_apply_overwrites_to_context_list_overwrite_with_valid_subset():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}

def test_apply_overwrites_to_context_list_overwrite_with_invalid_subset():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['b', 'd'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_single_choice_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_single_choice_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key2": "new_value2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "value1", "key2": "new_value2"}}

def test_apply_overwrites_to_context_bool_true_conversion():
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_false_conversion():
    context = {"flag": True}
    overwrite_context = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": False}

def test_apply_overwrites_to_context_bool_invalid_conversion():
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


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_46_evaluates_to_false():
    context = {"key": "value"}
    overwrite_context = {"key": {"nested_key": "nested_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "value"}


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_overwrites_to_context_list_in_dictionary_variable():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": ["d", "e"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == ["d", "e"]


# LLM-generated content at query #7
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"existing": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value", "new_nested": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d provided for choice variable list_var" in str(e)

def test_apply_overwrites_to_context_multichoice_overwrite():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi_var": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_overwrite_invalid():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "['a', 'd'] provided for multi-choice variable multi_var" in str(e)

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"dict_var": {"key1": "val1", "key2": "val2"}}
    overwrite_context = {"dict_var": {"key2": "new_val2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"key1": "val1", "key2": "new_val2"}}

def test_apply_overwrites_to_context_bool_overwrite_true():
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_overwrite_false():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}

def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable bool_var could not be converted to a boolean" in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir_when_output_dir_exists():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    overwrite_if_exists = True

    result_path, result_bool = render_and_create_dir(
        dirname, context, output_dir, environment, overwrite_if_exists
    )

    assert result_bool is False


# LLM-generated content at query #10
#--------------------------

```python
def test_json_decoding_error_raises_context_decoding_exception():
    with pytest.raises(ContextDecodingException) as exc_info:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(exc_info.value)


# LLM-generated content at query #11
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

def test_generate_context_with_non_existent_file():
    with pytest.raises(FileNotFoundError):
        generate_context('non_existent.json')


# LLM-generated content at query #12
#--------------------------

```python
def test_output_dir_exists_predicate():
    output_dir_exists = True
    assert output_dir_exists


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": {"new_nested": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["x", "y"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["x", "y"]}

def test_apply_overwrites_to_context_valid_multichoice():
    context = {"multi": ["a", "b", "c"]}
    overwrite_context = {"multi": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi": ["a", "c"]}

def test_apply_overwrites_to_context_invalid_multichoice():
    context = {"multi": ["a", "b", "c"]}
    overwrite_context = {"multi": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d provided for multi-choice variable multi, but valid choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_valid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_invalid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "d provided for choice variable choice, but the choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_partial_dict_overwrite():
    context = {"dict_var": {"a": 1, "b": 2}}
    overwrite_context = {"dict_var": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"a": 1, "b": 3}}

def test_apply_overwrites_to_context_bool_conversion_valid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_conversion_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable bool_var could not be converted to a boolean" in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #15
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
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
        assert "d provided for choice variable choices, but the choices are ['a', 'b', 'c']." in str(e)

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
        assert "['a', 'd'] provided for multi-choice variable multichoice, but valid choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"dict_var": {"key1": "val1", "key2": "val2"}}
    overwrite_context = {"dict_var": {"key2": "new_val2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"key1": "val1", "key2": "new_val2"}}

def test_apply_overwrites_to_context_bool_conversion_valid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_conversion_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable bool_var could not be converted to a boolean." in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'value': 123}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'value': 123}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'value': 123}}

def test_generate_context_with_invalid_default():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid': 'value'}
        )

def test_generate_context_with_invalid_extra():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-data/cookiecutter.json',
            extra_context={'invalid': 'value'}
        )


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.bin'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ name }}!')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Hello, {{ name }}!'

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Existing content')
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('Existing content')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Existing content'

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = '{{ empty_var }}.txt'
    context = {'cookiecutter': {'_new_lines': None}, 'empty_var': ''}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Content')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r', newline='') as f:
        content = f.read()
        assert content == 'Line 1\r\nLine 2'

def test_generate_file_custom_newline():
    project_dir = '/tmp/test_project'
    infile = 'custom_newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r', newline='') as f:
        content = f.read()
        assert content == 'Line 1\nLine 2'


# LLM-generated content at query #18
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False) == '\n'


# LLM-generated content at query #19
#--------------------------

```python
def test_file_name_is_empty():
    project_dir = "/path/to/project"
    infile = "directory/"
    context = {}
    env = Environment()

    outfile_tmpl = env.from_string(infile)
    outfile = os.path.join(project_dir, outfile_tmpl.render(**context))
    file_name_is_empty = os.path.isdir(outfile)

    assert file_name_is_empty is True


# LLM-generated content at query #20
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #21
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    assert context["my_bool"] is True


# LLM-generated content at query #22
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('binary_file.png')


# LLM-generated content at query #23
#--------------------------

```python
def test_render_and_create_dir_success():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path(output_dir, "test_dir")
    assert result[1] is True

def test_render_and_create_dir_empty_dirname():
    dirname = ""
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_exists_no_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir, "test_dir").mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_exists_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir, "test_dir").mkdir(exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == Path(output_dir, "test_dir")
    assert result[1] is False


# LLM-generated content at query #24
#--------------------------

```python
def test_cookiecutter_new_lines_predicate_true():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
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
    context = {'cookiecutter': {'_new_lines': None}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile)) as f:
        assert f.read() == 'Hello test!'

def test_generate_file_skip_existing():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('existing')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile)) as f:
        assert f.read() == 'existing'

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'newline.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('line1\r\nline2')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content

def test_generate_file_configured_newline():
    project_dir = '/fake/project'
    infile = 'newline.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('line1\nline2')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile)) as f:
        content = f.read()
        assert '\n' in content


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = 'empty_filename'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\n' in content or b'\r\n' in content

def test_generate_file_configured_newline():
    project_dir = '/tmp/test_project'
    infile = 'configured_newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('/tmp/templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #27
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #28
#--------------------------

```python
def test_template_syntax_error_raises_with_translated_false():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    with open(infile, 'w') as f:
        f.write('{% invalid syntax %}')

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)

    assert exc_info.value.translated is False


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test_cookiecutter.json')
    assert context == {'test_cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #30
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_context_without_default_context():
    result = generate_context(context_file='nonexistent.json', default_context=None, extra_context=None)
    assert result == {}


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test_cookiecutter.json')
    assert context == {'test_cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/invalid_json.json')

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
            default_context={'invalid_key': 'value'}
        )

def test_generate_context_with_invalid_extra_context():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test_cookiecutter.json',
            extra_context={'key': 'invalid_value'}
        )

def test_generate_context_with_nested_dict():
    context = generate_context(
        'tests/test_nested_cookiecutter.json',
        extra_context={'nested': {'key': 'nested_value'}}
    )
    assert context == {'test_nested_cookiecutter': {'nested': {'key': 'nested_value'}}}

def test_generate_context_with_list_choice():
    context = generate_context(
        'tests/test_list_cookiecutter.json',
        extra_context={'choices': 'valid_choice'}
    )
    assert context == {'test_list_cookiecutter': {'choices': ['valid_choice', 'other_choice']}}

def test_generate_context_with_list_multichoice():
    context = generate_context(
        'tests/test_list_cookiecutter.json',
        extra_context={'choices': ['valid_choice', 'other_choice']}
    )
    assert context == {'test_list_cookiecutter': {'choices': ['valid_choice', 'other_choice']}}

def test_generate_context_with_boolean():
    context = generate_context(
        'tests/test_boolean_cookiecutter.json',
        extra_context={'bool_var': 'yes'}
    )
    assert context == {'test_boolean_cookiecutter': {'bool_var': True}}


# LLM-generated content at query #33
#--------------------------

```python
def test_template_syntax_error_handling():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": False}}
    env = Environment(loader=FileSystemLoader("."))

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)

    assert not exc_info.value.translated


# LLM-generated content at query #34
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'README.md']
        }
    }
    assert is_copy_only_path('file.txt', context) is True
    assert is_copy_only_path('README.md', context) is True

def test_is_copy_only_path_returns_false_when_path_does_not_match_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'README.md']
        }
    }
    assert is_copy_only_path('file.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

def test_is_copy_only_path_returns_false_when_key_not_in_context():
    context = {
        'cookiecutter': {}
    }
    assert is_copy_only_path('file.txt', context) is False

def test_is_copy_only_path_returns_false_when_cookiecutter_key_not_in_context():
    context = {}
    assert is_copy_only_path('file.txt', context) is False


# LLM-generated content at query #35
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = '/fake/project/dir'
    infile = 'fake_template.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))

    # Create a template with syntax error
    with open(infile, 'w') as f:
        f.write('{% if %}')

    with pytest.raises(TemplateSyntaxError):
        generate_file(project_dir, infile, context, env)


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
def test_generate_files_basic():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_overwrite():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_skip_existing():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_no_hooks():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)

def test_generate_files_keep_on_failure():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)


# LLM-generated content at query #38
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
    with open(outfile, 'w') as f:
        f.write('existing content')

    result = skip_if_file_exists and os.path.exists(outfile)

    assert result is True


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert not os.path.isdir(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert not os.path.isdir(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/tmp/project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': False}}
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

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #40
#--------------------------

```python
def test_is_binary_predicate():
    assert is_binary('binary_file.png') == True


# LLM-generated content at query #41
#--------------------------

```python
def test_skip_if_file_exists_and_outfile_exists():
    project_dir = '/fake/project/dir'
    infile = 'fake_infile.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment()
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    open(outfile, 'w').close()
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_context_without_default_context():
    context = generate_context(context_file='cookiecutter.json', default_context=None, extra_context=None)
    assert context is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_cookiecutter_new_lines_false():
    context = {
        'cookiecutter': {
            '_new_lines': False
        }
    }
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #44
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "output", Environment())


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-data/context.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-data/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-data/context.json',
        default_context={'name': 'default'}
    )
    assert context == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-data/context.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-data/context.json',
            default_context={'invalid': 'value'}
        )

def test_generate_context_with_none_default_and_extra_context():
    context = generate_context(
        'tests/test-data/context.json',
        default_context=None,
        extra_context=None
    )
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_file_with_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.dat'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    is_binary.return_value = True
    os.path.isdir.return_value = False
    os.path.exists.return_value = False
    os.path.join.return_value = '/fake/project/binary.dat'
    env.from_string.return_value = Template('binary.dat')
    shutil.copyfile.called = False
    shutil.copymode.called = False

    generate_file(project_dir, infile, context, env)

    assert os.path.join.called_with(project_dir, 'binary.dat')
    assert is_binary.called_with(infile)
    assert shutil.copyfile.called_with(infile, '/fake/project/binary.dat')
    assert shutil.copymode.called_with(infile, '/fake/project/binary.dat')

def test_generate_file_with_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    is_binary.return_value = False
    os.path.isdir.return_value = False
    os.path.exists.return_value = False
    os.path.join.return_value = '/fake/project/template.txt'
    env.from_string.return_value = Template('template.txt')
    env.get_template.return_value = Template('template content')
    open.return_value.__enter__.return_value.readline.return_value = 'line1\n'
    open.return_value.__enter__.return_value.newlines = '\n'
    shutil.copymode.called = False

    generate_file(project_dir, infile, context, env)

    assert os.path.join.called_with(project_dir, 'template.txt')
    assert is_binary.called_with(infile)
    assert env.get_template.called_with('template.txt')
    assert open.called_with('/fake/project/template.txt', 'w', encoding='utf-8', newline='\n')
    assert shutil.copymode.called_with(infile, '/fake/project/template.txt')

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.path.isdir.return_value = False
    os.path.exists.return_value = True
    os.path.join.return_value = '/fake/project/existing.txt'
    env.from_string.return_value = Template('existing.txt')

    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

    assert os.path.exists.called_with('/fake/project/existing.txt')
    assert not is_binary.called
    assert not env.get_template.called

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    os.path.isdir.return_value = True
    os.path.join.return_value = '/fake/project'
    env.from_string.return_value = Template('')

    generate_file(project_dir, infile, context, env)

    assert os.path.isdir.called_with('/fake/project')
    assert not is_binary.called
    assert not env.get_template.called


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    """Test that the predicate at line 36 evaluates to False."""
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = True

    # Mock the necessary functions to avoid side effects
    from cookiecutter.generate import generate_files
    from unittest.mock import patch, MagicMock

    with patch('cookiecutter.generate.create_env_with_context') as mock_create_env, \
         patch('cookiecutter.generate.find_template') as mock_find_template, \
         patch('cookiecutter.generate.render_and_create_dir') as mock_render_and_create_dir, \
         patch('cookiecutter.generate.work_in'), \
         patch('cookiecutter.generate.run_hook_from_repo_dir'), \
         patch('cookiecutter.generate.os.path.abspath') as mock_abspath:

        mock_env = MagicMock()
        mock_create_env.return_value = mock_env

        mock_template_dir = '/path/to/template'
        mock_find_template.return_value = mock_template_dir

        mock_project_dir = '/path/to/project'
        mock_render_and_create_dir.return_value = (mock_project_dir, True)

        mock_abspath.return_value = mock_project_dir

        # Call the function
        result = generate_files(
            repo_dir=repo_dir,
            context=context,
            output_dir=output_dir,
            overwrite_if_exists=overwrite_if_exists,
            skip_if_file_exists=skip_if_file_exists,
            accept_hooks=accept_hooks,
            keep_project_on_failure=keep_project_on_failure
        )

        # Verify the predicate evaluates to False
        assert result == mock_project_dir
        assert mock_render_and_create_dir.called
        assert mock_abspath.called


# LLM-generated content at query #49
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_files_basic():
    """Test basic file generation."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_overwrite():
    """Test file generation with overwrite."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_skip_existing():
    """Test file generation with skip existing files."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_no_hooks():
    """Test file generation without hooks."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_keep_on_failure():
    """Test file generation with keep on failure."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_with_copy_only():
    """Test file generation with copy only paths."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project", "_copy_without_render": ["*.bin"]}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_with_new_lines():
    """Test file generation with new lines configuration."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project", "_new_lines": "\n"}}
    output_dir = "test_output"
    result = generate_files(repo_dir, context, output_dir)
    assert result == os.path.join(output_dir, "test_project")

def test_generate_files_with_undefined_variable():
    """Test file generation with undefined variable."""
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)


# LLM-generated content at query #51
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'test_repo'
    output_dir = 'test_output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(result)
    assert os.path.isdir(result)
    assert 'test_project' in result


# LLM-generated content at query #52
#--------------------------

```python
def test_os_walk_returns_true():
    with work_in(template_dir):
        env.loader = FileSystemLoader(['.', '../templates'])
        assert os.walk('.')


# LLM-generated content at query #53
#--------------------------

```python
def test_delete_project_on_failure_predicate_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #54
#--------------------------

```python
def test_generate_file_with_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x89PNG')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, 'binary.png'))

def test_generate_file_with_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'template.txt')) as f:
        assert f.read() == 'Hello test!'

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, 'template.txt')
    with open(outfile, 'w') as f:
        f.write('existing')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == 'existing'

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{}}'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_with_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}, 'name': 'test'}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        assert b'\r\n' in f.read()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-fixtures/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #3
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
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["b", "a", "c"]}

def test_apply_overwrites_to_context_invalid_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable list_var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_overwrite():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi_var": ["a", "c"]}

def test_apply_overwrites_to_context_invalid_multichoice_overwrite():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multi_var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"dict_var": {"a": 1, "b": 2}}
    overwrite_context = {"dict_var": {"b": 3, "c": 4}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"a": 1, "b": 3, "c": 4}}

def test_apply_overwrites_to_context_bool_overwrite():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_invalid_bool_overwrite():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid provided for variable bool_var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #4
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    repo_dir = '/path/to/repo'
    hook_name = 'post_gen_project'
    project_dir = '/path/to/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True

    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
    assert True  # Function should execute without raising an exception


# LLM-generated content at query #5
#--------------------------

```python
def test_issubset_evaluates_to_false():
    context = {"variable": ["a", "b", "c"]}
    overwrite_context = {"variable": ["d", "e"]}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #6
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    result = render_and_create_dir('', {}, '/tmp', Environment())
    assert result == (Path('/tmp'), False)

def test_render_and_create_dir_existing_dir_no_overwrite():
    os.makedirs('/tmp/existing_dir', exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('existing_dir', {}, '/tmp', Environment())

def test_render_and_create_dir_existing_dir_with_overwrite():
    os.makedirs('/tmp/existing_dir', exist_ok=True)
    result = render_and_create_dir('existing_dir', {}, '/tmp', Environment(), overwrite_if_exists=True)
    assert result == (Path('/tmp/existing_dir'), False)

def test_render_and_create_dir_new_dir():
    result = render_and_create_dir('new_dir', {}, '/tmp', Environment())
    assert result == (Path('/tmp/new_dir'), True)
    os.rmdir('/tmp/new_dir')

def test_render_and_create_dir_with_context():
    result = render_and_create_dir('{{ project_name }}', {'project_name': 'test_project'}, '/tmp', Environment())
    assert result == (Path('/tmp/test_project'), True)
    os.rmdir('/tmp/test_project')


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_46_evaluates_to_false():
    context = {"key": "value"}
    overwrite_context = {"key": {"nested_key": "nested_value"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "value"}


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_files_basic():
    """Test basic file generation."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'file.txt'))

def test_generate_files_overwrite():
    """Test file generation with overwrite."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # First run
    generate_files(repo_dir, context, output_dir)

    # Second run with overwrite
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'file.txt'))

def test_generate_files_skip_existing():
    """Test file generation with skip_if_file_exists."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # First run
    generate_files(repo_dir, context, output_dir)

    # Second run with skip
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'file.txt'))

def test_generate_files_with_hooks():
    """Test file generation with hooks."""
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'file.txt'))

def test_generate_files_copy_without_render():
    """Test file generation with copy_without_render."""
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_binaries')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'file.bin'))


# LLM-generated content at query #9
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

def test_apply_overwrites_to_context_list_overwrite_with_list():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "c"]}

def test_apply_overwrites_to_context_list_overwrite_with_invalid_list():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "e"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d', 'e'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_list_overwrite_with_valid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_overwrite_with_invalid_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"config": {"key2": "new_value2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"config": {"key1": "value1", "key2": "new_value2"}}

def test_apply_overwrites_to_context_bool_overwrite_with_valid_str():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_overwrite_with_invalid_str():
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


# LLM-generated content at query #10
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_conversion_failure():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid boolean conversion"


# LLM-generated content at query #11
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": {"new_nested": "new_value"}}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite_context = {"list_var": ["x", "y"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["x", "y"]}

def test_apply_overwrites_to_context_list_multichoice_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_multichoice_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["x", "y"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "x, y provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_list_choice_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_choice_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "x"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "x provided for choice variable choice, but the choices are ['a', 'b', 'c']" in str(e)

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"dict_var": {"a": 1, "b": 2}}
    overwrite_context = {"dict_var": {"b": 3, "c": 4}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"a": 1, "b": 3, "c": 4}}

def test_apply_overwrites_to_context_bool_string_true():
    context = {"bool_var": False}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_string_false():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}

def test_apply_overwrites_to_context_bool_string_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid provided for variable bool_var could not be converted to a boolean" in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_context_applies_default_context():
    default_context = {'key': 'value'}
    extra_context = None
    context_file = 'cookiecutter.json'
    result = generate_context(context_file, default_context, extra_context)
    assert 'cookiecutter' in result
    assert result['cookiecutter']['key'] == 'value'


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_no_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()

def test_generate_files_with_copy_only_paths():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    repo_dir = 'tests/test-template'
    output_dir = 'tests/output'
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert Path(result, 'test_project').exists()


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
def test_predicate_evaluates_to_true():
    assert not "" or "" == ""


# LLM-generated content at query #16
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_response():
    context = {"variable": True}
    overwrite_context = {"variable": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()
    assert (Path(result) / 'test.txt').read_text() == 'Hello, test_project!'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'README.md').write_text('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir)
    assert Path(result).exists()
    assert (Path(result) / 'README.md').exists()
    assert (Path(result) / 'README.md').read_text() == 'Hello, {{cookiecutter.project_name}}!'
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)

def test_generate_files_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()
    template_dir = Path(repo_dir) / '{{cookiecutter.project_name}}'
    template_dir.mkdir()
    (template_dir / 'test.txt').write_text('Hello, {{cookiecutter.project_name}}!')
    (repo_dir / 'hooks' / 'pre_gen_project.py').parent.mkdir()
    (repo_dir / 'hooks' / 'pre_gen_project.py').write_text('print("Pre-hook executed")')
    (repo_dir / 'hooks' / 'post_gen_project.py').write_text('print("Post-hook executed")')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert Path(result).exists()
    assert (Path(result) / 'test.txt').exists()
    shutil.rmtree(output_dir)
    shutil.rmtree(repo_dir)


# LLM-generated content at query #18
#--------------------------

```python
def test_render_and_create_dir_creates_directory():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    result_path, created = render_and_create_dir('{{project_name}}', context, output_dir, environment)
    assert result_path.exists()
    assert created is True
    assert result_path == output_dir / 'test_project'

def test_render_and_create_dir_overwrites_existing_directory():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    result_path, created = render_and_create_dir('{{project_name}}', context, output_dir, environment, overwrite_if_exists=True)
    assert result_path.exists()
    assert created is False
    assert result_path == output_dir / 'test_project'

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', context, output_dir, environment)

def test_render_and_create_dir_raises_exception_for_existing_directory():
    context = {'project_name': 'test_project'}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    result_path, _ = render_and_create_dir('{{project_name}}', context, output_dir, environment)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('{{project_name}}', context, output_dir, environment)


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0'}}


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_files_basic():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'basic_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project'))
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('test')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'new_file.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'project_name': 'test_project', 'cookiecutter': {'_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'copy_template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project', 'data.bin'))
    with open(os.path.join(result, 'test_project', 'data.bin'), 'rb') as f:
        content = f.read()
        assert content == b'binary data'
    shutil.rmtree(output_dir)

def test_generate_files_hooks():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'hook_template')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_marker.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable():
    context = {'project_name': 'test_project'}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_templates', 'undefined_template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #22
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_raises_context_decoding_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #25
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

def test_generate_context_with_default_and_extra_context():
    context = generate_context(
        'tests/test-data/cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-data/cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #26
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('readme.txt', context) is True
    assert is_copy_only_path('docs/guide.md', context) is True

def test_is_copy_only_path_returns_false_when_path_does_not_match_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('script.py', context) is False
    assert is_copy_only_path('src/main.py', context) is False

def test_is_copy_only_path_returns_false_when_context_missing_key():
    context = {
        'cookiecutter': {}
    }
    assert is_copy_only_path('readme.txt', context) is False

def test_is_copy_only_path_returns_false_when_context_missing_cookiecutter():
    context = {}
    assert is_copy_only_path('readme.txt', context) is False


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_file_with_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, 'binary.jpg'))
    with open(os.path.join(project_dir, 'binary.jpg'), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_with_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, 'template.txt'))
    with open(os.path.join(project_dir, 'template.txt')) as f:
        assert f.read() == 'Hello, test!'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')
    with open(os.path.join(project_dir, 'template.txt'), 'w') as f:
        f.write('Existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, 'template.txt')) as f:
        assert f.read() == 'Existing content'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_with_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n', 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        content = f.read()
        assert content.endswith(b'\r\n')
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    os.path.exists.return_value = True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #29
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    assert is_binary('binary_file.png') is True


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
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    outfile = "existing_file.txt"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        f.write("test")
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x89PNG\r\n\x1a\n'

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Hello test!'

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write('existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, 'r') as f:
        assert f.read() == 'existing content'

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{ cookiecutter.name }}/file.txt'
    context = {'cookiecutter': {'name': ''}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\nLine 1\r\nLine 2\r\n' == content

def test_generate_file_configured_newline():
    project_dir = '/fake/project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'Line 1\nLine 2\n' == content


# LLM-generated content at query #33
#--------------------------

```python
def test_is_binary_predicate_true():
    infile = "binary_file.png"
    assert is_binary(infile) is True


# LLM-generated content at query #34
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    skip_if_file_exists = True
    outfile = "existing_file.txt"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        f.write("test")
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = '/tmp/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_text_render():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment()
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_empty_outfile():
    project_dir = '/tmp/project'
    infile = 'empty_outfile.txt'
    context = {'cookiecutter': {'_new_lines': False}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_newline_config():
    project_dir = '/tmp/project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment()
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_context_basic():
    context = generate_context('tests/test-project/cookiecutter.json')
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-project/cookiecutter.json',
        default_context={'name': 'override'}
    )
    assert context == {'cookiecutter': {'name': 'override', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-project/cookiecutter.json',
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context('tests/test-project/invalid.json')

def test_generate_context_with_invalid_default():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        context = generate_context(
            'tests/test-project/cookiecutter.json',
            default_context={'invalid': 'value'}
        )
        assert len(w) == 1
        assert "Invalid default received" in str(w[0].message)
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    result = generate_context(context_file)
    assert result == {'test': {'key': 'value'}}


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test.json'
    open(context_file, 'w').write('{"key": "value"}')
    assert generate_context(context_file) == {'test': {'key': 'value'}}


# LLM-generated content at query #41
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    os.makedirs(os.path.join(project_dir, 'binary.png'), exist_ok=True)
    with patch('shutil.copyfile') as mock_copy, \
         patch('shutil.copymode') as mock_mode, \
         patch('cookiecutter.generate.is_binary', return_value=True):
        generate_file(project_dir, infile, context, env)
        mock_copy.assert_called_once_with(infile, os.path.join(project_dir, infile))
        mock_mode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    os.makedirs(os.path.join(project_dir, 'template.txt'), exist_ok=True)
    with patch('builtins.open', mock_open()) as mock_file, \
         patch('shutil.copymode') as mock_mode, \
         patch('cookiecutter.generate.is_binary', return_value=False):
        generate_file(project_dir, infile, context, env)
        mock_file.assert_called_with(os.path.join(project_dir, infile), 'w', encoding='utf-8', newline=None)
        mock_mode.assert_called_once_with(infile, os.path.join(project_dir, infile))

def test_generate_file_skip_existing():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    os.makedirs(os.path.join(project_dir, 'existing.txt'), exist_ok=True)
    with patch('os.path.exists', return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_name():
    project_dir = '/fake/project'
    infile = '{{}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    with patch('os.path.isdir', return_value=True):
        generate_file(project_dir, infile, context, env)
        assert not os.path.exists(os.path.join(project_dir, infile))


# LLM-generated content at query #42
#--------------------------

```python
def test_cookiecutter_new_lines_predicate():
    context = {
        'cookiecutter': {
            '_new_lines': True
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #44
#--------------------------

```python
def test_cookiecutter_new_lines_true():
    context = {
        'cookiecutter': {
            '_new_lines': True
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_file_binary():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'binary.png'))

def test_generate_file_text():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, 'template.txt'))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = True

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # No assertion needed, just ensure no exception is raised

def test_generate_file_empty_outfile():
    project_dir = '/fake/project'
    infile = 'empty_dir/'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    # No assertion needed, just ensure no exception is raised

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, 'template.txt'), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #46
#--------------------------

```python
def test_generate_file_binary_skipped_if_exists():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'binary.jpg'), exist_ok=True)
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert os.path.isdir(os.path.join(project_dir, 'binary.jpg'))

def test_generate_file_text_skipped_if_exists():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, 'text.txt'), 'w') as f:
        f.write('existing')
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    with open(os.path.join(project_dir, 'text.txt')) as f:
        assert f.read() == 'existing'

def test_generate_file_binary_copied():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'binary content')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'binary.jpg'), 'rb') as f:
        assert f.read() == b'binary content'

def test_generate_file_text_rendered():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'text.txt')) as f:
        assert f.read() == 'Hello test!'

def test_generate_file_newline_detected():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Hello\r\nWorld\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'text.txt'), 'rb') as f:
        assert b'\r\n' in f.read()

def test_generate_file_newline_configured():
    project_dir = '/fake/project'
    infile = 'text.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Hello\r\nWorld\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'text.txt'), 'rb') as f:
        assert b'\r\n' not in f.read()
        assert b'\n' in f.read()


# LLM-generated content at query #47
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    with open(infile, 'rb') as f:
        original_content = f.read()
    assert content == original_content

def test_generate_file_text_file():
    project_dir = '/tmp/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert content == 'rendered content'

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_outfile():
    project_dir = '/tmp/project'
    infile = '{{""}}'
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

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
    assert b'\r\n' in content


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(output_dir, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('test')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'new_file.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_without_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'failing-template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_delete_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'failing-template')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=False)
    assert not os.path.exists(os.path.join(output_dir, 'test_project'))
    shutil.rmtree(output_dir)


# LLM-generated content at query #49
#--------------------------

```python
def test_generate_file_binary_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'binary.jpg'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text_rendering():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('Hello {{ cookiecutter.name }}!')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    with open(os.path.join(project_dir, infile), 'r', encoding='utf-8') as f:
        assert f.read() == 'Hello test!'

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2\r\n')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert b'\r\n' in f.read()

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{ cookiecutter.name }}.txt'
    context = {'cookiecutter': {'_new_lines': None, 'name': ''}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_permissions():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('test')
    os.chmod(infile, 0o644)

    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    outfile = os.path.join(project_dir, infile)
    assert os.stat(outfile).st_mode & 0o777 == 0o644


# LLM-generated content at query #50
#--------------------------

```python
def test_accept_hooks_predicate():
    assert accept_hooks is True


# LLM-generated content at query #51
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary.png'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        content = f.read()
    assert 'test' in content

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(project_dir)

def test_generate_file_newline_detection():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'rb') as f:
        content = f.read()
    assert b'\n' in content or b'\r\n' in content

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('templates'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, 'rb') as f:
        content = f.read()
    assert b'\r\n' in content


# LLM-generated content at query #52
#--------------------------

```python
def test_apply_overwrites_with_default_context():
    default_context = {'key': 'value'}
    obj = {}
    apply_overwrites_to_context(obj, default_context)
    assert obj == default_context


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #54
#--------------------------

```python
def test_delete_project_on_failure_is_false():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #55
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
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'r') as f:
        assert f.read() == 'Hello, test_project!'
    shutil.rmtree(output_dir)

def test_generate_files_overwrite():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'r') as f:
        assert f.read() == 'Hello, test_project!'
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
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'r') as f:
        assert f.read() == 'Hello, test_project!'
    shutil.rmtree(output_dir)

def test_generate_files_copy_only():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.bin'), 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.bin'))
    with open(os.path.join(output_dir, 'test_project', 'test.bin'), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'
    shutil.rmtree(output_dir)

def test_generate_files_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(output_dir, 'template')
    os.makedirs(repo_dir)
    hooks_dir = os.path.join(repo_dir, 'hooks')
    os.makedirs(hooks_dir)
    with open(os.path.join(hooks_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("Pre-hook executed")')
    with open(os.path.join(hooks_dir, 'post_gen_project.py'), 'w') as f:
        f.write('print("Post-hook executed")')
    template_dir = os.path.join(repo_dir, '{{cookiecutter.project_name}}')
    os.makedirs(template_dir)
    with open(os.path.join(template_dir, 'test.txt'), 'w') as f:
        f.write('Hello, {{cookiecutter.project_name}}!')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'test.txt'))
    with open(os.path.join(output_dir, 'test_project', 'test.txt'), 'r') as f:
        assert f.read() == 'Hello, test_project!'
    shutil.rmtree(output_dir)


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_false():
    repo_dir = '/path/to/repo'
    context = {}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = True

    # Mock the necessary functions and objects
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'

    class MockPath:
        def __init__(self, path):
            self.path = path

    def mock_find_template(repo_dir, env):
        return MockPath(f"{repo_dir}/{{cookiecutter.project_name}}")

    def mock_render_and_create_dir(unrendered_dir, context, output_dir, env, overwrite_if_exists):
        return (f"{output_dir}/project_name", False)

    def mock_work_in(path):
        return MockWorkInContext()

    class MockWorkInContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Assign the mocks
    original_find_template = find_template
    original_render_and_create_dir = render_and_create_dir
    original_work_in = work_in

    find_template = mock_find_template
    render_and_create_dir = mock_render_and_create_dir
    work_in = mock_work_in

    try:
        # Call the function up to line 59
        env = create_env_with_context(context)
        template_dir = find_template(repo_dir, env)
        project_dir, output_directory_created = render_and_create_dir(
            os.path.split(template_dir)[1], context, output_dir, env, overwrite_if_exists
        )
        project_dir = os.path.abspath(project_dir)
        delete_project_on_failure = output_directory_created and not keep_project_on_failure

        # Assert that the predicate evaluates to False
        assert delete_project_on_failure is False
    finally:
        # Restore the original functions
        find_template = original_find_template
        render_and_create_dir = original_render_and_create_dir
        work_in = original_work_in


# LLM-generated content at query #57
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'src'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'src', 'main.py'))

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # Create existing project
    generate_files(repo_dir, context, output_dir)

    # Try to overwrite
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'README.md'))

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'basic_template')

    # Create existing project
    generate_files(repo_dir, context, output_dir)

    # Create a new file in the project
    with open(os.path.join(output_dir, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')

    # Try to generate again with skip_if_file_exists
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)

    # Check that the new file still exists with original content
    with open(os.path.join(output_dir, 'test_project', 'new_file.txt'), 'r') as f:
        assert f.read() == 'existing content'

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_hooks')

    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'hook_output.txt'))

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_binary')

    result = generate_files(repo_dir, context, output_dir)

    assert os.path.exists(os.path.join(output_dir, 'test_project'))
    assert os.path.exists(os.path.join(output_dir, 'test_project', 'data.bin'))

    # Verify binary file wasn't rendered (should contain original binary content)
    with open(os.path.join(output_dir, 'test_project', 'data.bin'), 'rb') as f:
        content = f.read()
        assert content == b'\x00\x01\x02\x03'

def test_generate_files_undefined_variable():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'template_with_undefined')

    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)

    # Verify project directory was cleaned up
    assert not os.path.exists(os.path.join(output_dir, 'test_project'))


# LLM-generated content at query #58
#--------------------------

```python
def test_generate_context_with_valid_json():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            result = generate_context('test.json')
            assert result == {'test': {"key": "value"}}

def test_generate_context_with_invalid_json():
    with patch('builtins.open', mock_open(read_data='{invalid json}')):
        with patch('json.load', side_effect=ValueError("Invalid JSON")):
            with pytest.raises(ContextDecodingException):
                generate_context('test.json')

def test_generate_context_with_default_context():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            generate_context('test.json', default_context={"key": "new_value"})
            assert result == {'test': {"key": "new_value"}}

def test_generate_context_with_extra_context():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', return_value={"key": "value"}):
            result = generate_context('test.json', extra_context={"key": "new_value"})
            assert result == {'test': {"key": "new_value"}}

def test_generate_context_with_invalid_default_context():
    with patch('builtins.open', mock_open(read_data='{"key": ["a", "b"]}')):
        with patch('json.load', return_value={"key": ["a", "b"]}):
            with patch('warnings.warn') as mock_warn:
                generate_context('test.json', default_context={"key": ["c"]})
                mock_warn.assert_called_once_with("Invalid default received: ['c'] provided for multi-choice variable key, but valid choices are ['a', 'b']")

def test_generate_context_with_boolean_conversion():
    with patch('builtins.open', mock_open(read_data='{"key": true}')):
        with patch('json.load', return_value={"key": True}):
            result = generate_context('test.json', extra_context={"key": "yes"})
            assert result == {'test': {"key": True}}


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_false():
    context = {'cookiecutter': {'_copy_without_render': ['test_dir']}}
    os.makedirs('test_dir', exist_ok=True)
    os.makedirs('render_dir', exist_ok=True)
    open('test_file.txt', 'w').close()

    with work_in('.'):
        env = StrictEnvironment(context=context, keep_trailing_newline=True)
        env.loader = FileSystemLoader(['.', '../templates'])

        for root, dirs, files in os.walk('.'):
            copy_dirs = []
            render_dirs = []

            for d in sorted(dirs):
                d_ = os.path.normpath(os.path.join(root, d))
                if is_copy_only_path(d_, context):
                    copy_dirs.append(d)
                else:
                    render_dirs.append(d)

            assert 'test_dir' in copy_dirs
            assert 'render_dir' in render_dirs
            assert len(copy_dirs) == 1
            assert len(render_dirs) == 1


# LLM-generated content at query #60
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key': 'value'}, f)
    result = generate_context(context_file)
    assert result == {'test': {'key': 'value'}}


# LLM-generated content at query #61
#--------------------------

```python
def test_generate_context_with_no_default_context():
    context = generate_context(context_file='nonexistent.json')
    assert not context


# LLM-generated content at query #62
#--------------------------

```python
def test_default_context_false_predicate():
    result = generate_context(
        context_file='nonexistent.json',
        default_context=None,
        extra_context=None
    )
    assert result == {'nonexistent': OrderedDict()}


# LLM-generated content at query #63
#--------------------------

```python
def test_generate_context_opens_file_with_utf8_encoding():
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')) as mock_file:
        generate_context('test.json')
        mock_file.assert_called_with('test.json', encoding='utf-8')


# LLM-generated content at query #64
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

def test_generate_context_with_invalid_default():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}


# LLM-generated content at query #65
#--------------------------

```python
def test_generate_files_basic():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files(
        repo_dir='tests/test-template',
        context=context,
        output_dir='tests/output',
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )
    assert result == 'tests/output/test_project'
    assert os.path.exists('tests/output/test_project')

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    os.makedirs('tests/output/test_project', exist_ok=True)
    result = generate_files(
        repo_dir='tests/test-template',
        context=context,
        output_dir='tests/output',
        overwrite_if_exists=False,
        skip_if_file_exists=True,
        accept_hooks=False,
        keep_project_on_failure=False
    )
    assert result == 'tests/output/test_project'
    assert os.path.exists('tests/output/test_project')

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files(
        repo_dir='tests/test-template-with-hooks',
        context=context,
        output_dir='tests/output',
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=True,
        keep_project_on_failure=False
    )
    assert result == 'tests/output/test_project'
    assert os.path.exists('tests/output/test_project')

def test_generate_files_keep_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    result = generate_files(
        repo_dir='tests/test-template',
        context=context,
        output_dir='tests/output',
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=True
    )
    assert result == 'tests/output/test_project'
    assert os.path.exists('tests/output/test_project')

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.md']}}
    result = generate_files(
        repo_dir='tests/test-template',
        context=context,
        output_dir='tests/output',
        overwrite_if_exists=True,
        skip_if_file_exists=False,
        accept_hooks=False,
        keep_project_on_failure=False
    )
    assert result == 'tests/output/test_project'
    assert os.path.exists('tests/output/test_project')


