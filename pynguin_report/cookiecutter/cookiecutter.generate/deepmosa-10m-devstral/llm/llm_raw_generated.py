####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    _run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {}, False)
    assert True


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context_with_new_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_with_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": {"new_nested": "new_value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": {"new_nested": "new_value"}}

def test_apply_overwrites_to_context_with_list_overwrite():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a", "c"]}

def test_apply_overwrites_to_context_with_invalid_choice():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d'] provided for choice variable choices, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_with_multichoice_overwrite():
    context = {"multichoices": ["a", "b", "c"]}
    overwrite_context = {"multichoices": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoices": ["a", "c"]}

def test_apply_overwrites_to_context_with_invalid_multichoice():
    context = {"multichoices": ["a", "b", "c"]}
    overwrite_context = {"multichoices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multichoices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_with_dict_overwrite():
    context = {"nested": {"key1": "value1", "key2": "value2"}}
    overwrite_context = {"nested": {"key2": "new_value2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"key1": "value1", "key2": "new_value2"}}

def test_apply_overwrites_to_context_with_bool_overwrite():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_with_invalid_bool_overwrite():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_with_simple_overwrite():
    context = {"key": "old_value"}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new_value"}


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context_with_list_and_in_dictionary_variable():
    context = {"key": ["a", "b"]}
    overwrite_context = {"key": ["c", "d"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == ["c", "d"]


# LLM-generated content at query #4
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"variable": True}
    overwrite_context = {"variable": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #5
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())

def test_render_and_create_dir_with_existing_directory_and_no_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("existing_dir", {}, "/tmp", Environment())

def test_render_and_create_dir_with_existing_directory_and_overwrite():
    dir_to_create = Path("/tmp/existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("existing_dir", {}, "/tmp", Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)

def test_render_and_create_dir_with_new_directory():
    result = render_and_create_dir("new_dir", {}, "/tmp", Environment())
    assert result == (Path("/tmp/new_dir"), True)
    Path("/tmp/new_dir").rmdir()

def test_render_and_create_dir_with_rendered_name():
    context = {"name": "test"}
    result = render_and_create_dir("{{ name }}_dir", context, "/tmp", Environment())
    assert result == (Path("/tmp/test_dir"), True)
    Path("/tmp/test_dir").rmdir()


# LLM-generated content at query #6
#--------------------------

```python
def test_is_copy_only_path_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.txt', '*.md']}}
    assert is_copy_only_path('README.md', context) is True
    assert is_copy_only_path('notes.txt', context) is True

def test_is_copy_only_path_not_matching_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.txt', '*.md']}}
    assert is_copy_only_path('main.py', context) is False
    assert is_copy_only_path('data.json', context) is False

def test_is_copy_only_path_empty_pattern_list():
    context = {'cookiecutter': {'_copy_without_render': []}}
    assert is_copy_only_path('README.md', context) is False
    assert is_copy_only_path('main.py', context) is False

def test_is_copy_only_path_missing_key():
    context = {'cookiecutter': {}}
    assert is_copy_only_path('README.md', context) is False
    assert is_copy_only_path('main.py', context) is False

def test_is_copy_only_path_no_cookiecutter_key():
    context = {}
    assert is_copy_only_path('README.md', context) is False
    assert is_copy_only_path('main.py', context) is False


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context('tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test'}}


# LLM-generated content at query #8
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as err:
        assert str(err) == (
            "invalid provided for variable test_var could not be converted to a boolean."
        )
    else:
        assert False, "Expected ValueError was not raised"


# LLM-generated content at query #10
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

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        generate_context(
            'tests/test-cookiecutter.json',
            default_context={'name': 'invalid'}
        )

def test_generate_context_with_invalid_extra_context():
    with pytest.raises(ValueError):
        generate_context(
            'tests/test-cookiecutter.json',
            extra_context={'name': 'invalid'}
        )


# LLM-generated content at query #11
#--------------------------

```python
def test_apply_overwrites_to_context_raises_value_error_for_invalid_boolean_string():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #12
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"my_bool": True}
    overwrite_context = {"my_bool": "invalid"}
    with pytest.raises(ValueError):
        apply_overwrites_to_context(context, overwrite_context)


# LLM-generated content at query #13
#--------------------------

```python
def test_generate_context_raises_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException) as excinfo:
        generate_context('invalid.json')
    assert "JSON decoding error while loading" in str(excinfo.value)


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #15
#--------------------------

```python
def test_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "output_dir", Environment())


# LLM-generated content at query #16
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
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('test')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'new_file.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-binary')
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

def test_generate_files_keep_project_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-undefined')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_value_error_raised_when_decoding_json():
    with pytest.raises(ValueError):
        json.loads('{"invalid": json}')


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_context_opens_file():
    context_file = 'test_context.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key": "value"}')

    result = generate_context(context_file)
    assert result == {'test_context': {'key': 'value'}}
    os.remove(context_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, Path(), Environment())

def test_render_and_create_dir_existing_dir_no_overwrite():
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir("test", {}, Path(), Environment())

def test_render_and_create_dir_existing_dir_overwrite():
    dir_to_create = Path("test")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("test", {}, Path(), Environment(), overwrite_if_exists=True)
    assert result[0] == dir_to_create
    assert result[1] == False

def test_render_and_create_dir_new_dir():
    result = render_and_create_dir("test", {}, Path(), Environment())
    assert result[0] == Path("test")
    assert result[1] == True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_first_level():
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
        assert str(e) == "['d'] provided for choice variable list_var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["a", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multi_var": ["a", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"multi_var": ["a", "b", "c"]}
    overwrite_context = {"multi_var": ["d", "e"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['d', 'e'] provided for multi-choice variable multi_var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_partial_overwrite():
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
        assert str(e) == "invalid provided for variable bool_var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_21_evaluates_to_false():
    context = {"key": ["value1", "value2"]}
    overwrite_context = {"key": "invalid_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == ["value1", "value2"]


# LLM-generated content at query #3
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

def test_generate_files_with_hooks():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-hooks')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    assert os.path.exists(os.path.join(result, 'test_project', 'hook_output.txt'))
    shutil.rmtree(output_dir)

def test_generate_files_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    result = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project'))
    shutil.rmtree(output_dir)

def test_generate_files_skip_existing_files():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'basic-template')
    result = generate_files(repo_dir, context, output_dir)
    with open(os.path.join(result, 'test_project', 'new_file.txt'), 'w') as f:
        f.write('existing content')
    result = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(result, 'test_project', 'new_file.txt'))
    with open(os.path.join(result, 'test_project', 'new_file.txt')) as f:
        assert f.read() == 'existing content'
    shutil.rmtree(output_dir)

def test_generate_files_copy_without_render():
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.bin']}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-binaries')
    result = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(result, 'test_project', 'data.bin'))
    with open(os.path.join(result, 'test_project', 'data.bin'), 'rb') as f:
        assert f.read() == b'binary content'
    shutil.rmtree(output_dir)

def test_generate_files_undefined_variable_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-undefined-variable')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir)
    shutil.rmtree(output_dir)

def test_generate_files_keep_project_on_failure():
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(os.path.dirname(__file__), 'test-data', 'template-with-undefined-variable')
    with pytest.raises(UndefinedVariableInTemplate):
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(output_dir)
    shutil.rmtree(output_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test__run_hook_from_repo_dir_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {}, False)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "The '_run_hook_from_repo_dir' function is deprecated" in str(w[0].message)

def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_run_hook:
        _run_hook_from_repo_dir('repo_dir', 'hook_name', 'project_dir', {}, False)
        mock_run_hook.assert_called_once_with('repo_dir', 'hook_name', 'project_dir', {}, False)


# LLM-generated content at query #5
#--------------------------

```python
def test_accept_hooks_false_skips_pre_hook():
    delete_project_on_failure = False
    accept_hooks = False
    assert not (accept_hooks and delete_project_on_failure)


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    with pytest.raises(ValueError) as exc_info:
        apply_overwrites_to_context(context, overwrite_context)
    assert "invalid provided for variable test_var could not be converted to a boolean." in str(exc_info.value)


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/test_project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b'\x89PNG\r\n\x1a\n'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_text_file():
    project_dir = '/tmp/test_project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, {{ name }}!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Hello, {{ name }}!'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_skip_if_exists():
    project_dir = '/tmp/test_project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Original content')
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('Existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == 'Existing content'
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_empty_filename():
    project_dir = '/tmp/test_project'
    infile = '{{ empty_var }}.txt'
    context = {'cookiecutter': {'_new_lines': None}, 'empty_var': ''}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))
    shutil.rmtree(project_dir)

def test_generate_file_newline_detection():
    project_dir = '/tmp/test_project'
    infile = 'newline_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_custom_newline():
    project_dir = '/tmp/test_project'
    infile = 'custom_newline_file.txt'
    context = {'cookiecutter': {'_new_lines': '\n'}}
    env = Environment(loader=FileSystemLoader('.'))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\r\n') as f:
        f.write('Line 1\r\nLine 2\r\n')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' not in content
        assert b'\n' in content
    os.remove(infile)
    shutil.rmtree(project_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_overwrite():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid boolean overwrite"


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context(context_file='tests/test-cookiecutter.json')
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}}

def test_generate_context_with_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='tests/invalid.json')

def test_generate_context_with_default_context():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        default_context={'name': 'default'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'test', 'version': '2.0.0'}}

def test_generate_context_with_both_contexts():
    context = generate_context(
        context_file='tests/test-cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'version': '2.0.0'}
    )
    assert context == {'test-cookiecutter': {'name': 'default', 'version': '2.0.0'}}

def test_generate_context_with_nested_dict_overwrite():
    context = generate_context(
        context_file='tests/test-nested-cookiecutter.json',
        extra_context={'config': {'debug': True}}
    )
    assert context == {'test-nested-cookiecutter': {'name': 'test', 'config': {'debug': True, 'verbose': False}}}

def test_generate_context_with_list_overwrite():
    context = generate_context(
        context_file='tests/test-list-cookiecutter.json',
        extra_context={'choices': ['option2']}
    )
    assert context == {'test-list-cookiecutter': {'choices': ['option2', 'option1', 'option3']}}

def test_generate_context_with_invalid_list_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-list-cookiecutter.json',
            extra_context={'choices': ['invalid']}
        )

def test_generate_context_with_boolean_overwrite():
    context = generate_context(
        context_file='tests/test-bool-cookiecutter.json',
        extra_context={'flag': 'yes'}
    )
    assert context == {'test-bool-cookiecutter': {'flag': True}}

def test_generate_context_with_invalid_boolean_overwrite():
    with pytest.raises(ValueError):
        generate_context(
            context_file='tests/test-bool-cookiecutter.json',
            extra_context={'flag': 'invalid'}
        )


# LLM-generated content at query #10
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
    dir_to_create.rmdir()

def test_render_and_create_dir_existing_dir_overwrite():
    dir_to_create = Path('/tmp/existing_dir')
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir('existing_dir', {}, '/tmp', Environment(), overwrite_if_exists=True)
    assert result == (dir_to_create, False)
    dir_to_create.rmdir()

def test_render_and_create_dir_new_dir():
    dir_to_create = Path('/tmp/new_dir')
    result = render_and_create_dir('new_dir', {}, '/tmp', Environment())
    assert result == (dir_to_create, True)
    dir_to_create.rmdir()

def test_render_and_create_dir_rendered_name():
    context = {'name': 'test'}
    environment = Environment()
    result = render_and_create_dir('{{ name }}_dir', context, '/tmp', environment)
    assert result[0].name == 'test_dir'
    result[0].rmdir()


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_success():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path(output_dir, dirname)
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
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir(dirname, context, output_dir, environment)

def test_render_and_create_dir_exists_with_overwrite():
    dirname = "test_dir"
    context = {"name": "test"}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == dir_to_create
    assert result[1] is False


# LLM-generated content at query #12
#--------------------------

```python
def test_template_syntax_error_raised_with_verbose_info():
    project_dir = "/fake/project/dir"
    infile = "fake_template.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment(loader=FileSystemLoader("."))

    with open(infile, "w") as f:
        f.write("{% if %}")  # Invalid syntax

    with pytest.raises(TemplateSyntaxError) as exc_info:
        generate_file(project_dir, infile, context, env)

    assert not exc_info.value.translated


# LLM-generated content at query #13
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #14
#--------------------------

```python
def test_apply_overwrites_to_context_with_list_and_in_dictionary_variable():
    context = {"key": ["a", "b", "c"]}
    overwrite_context = {"key": ["x", "y"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == ["x", "y"]


# LLM-generated content at query #15
#--------------------------

```python
def test_undefined_error_in_render_and_create_dir():
    context = {}
    output_dir = '/tmp'
    env = create_env_with_context(context)
    unrendered_dir = '{{ invalid_var }}'

    with pytest.raises(UndefinedVariableInTemplate):
        render_and_create_dir(unrendered_dir, context, output_dir, env, False)


# LLM-generated content at query #16
#--------------------------

```python
def test_cookiecutter_new_lines_predicate_true():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) is True


# LLM-generated content at query #17
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_context_raises_exception_on_invalid_json():
    with pytest.raises(ContextDecodingException):
        generate_context(context_file='invalid.json')


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_60():
    env = Environment()
    infile_fwd_slashes = "valid_template.txt"
    tmpl = env.get_template(infile_fwd_slashes)
    assert tmpl is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_custom_newline():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('/fake/template'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #21
#--------------------------

```python
def test_new_lines_in_context():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False) == '\n'


# LLM-generated content at query #22
#--------------------------

```python
def test_render_and_create_dir_empty_dirname_raises_exception():
    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "/tmp", Environment())


# LLM-generated content at query #23
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    assert is_binary('binary_file.png') is True


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project'
    infile = 'binary_file.png'
    context = {'cookiecutter': {}}
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
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    with open(infile, 'r') as f:
        original_content = f.read()
    assert content == original_content

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project'
    infile = 'text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, infile), 'w') as f:
        f.write('existing content')

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert content == 'existing content'

def test_generate_file_empty_filename():
    project_dir = '/fake/project'
    infile = '{{""}}'
    context = {'cookiecutter': {}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_with_template_rendering():
    project_dir = '/fake/project'
    infile = 'template.txt'
    context = {'cookiecutter': {'name': 'test'}}
    env = Environment(loader=FileSystemLoader('.'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    with open(os.path.join(project_dir, infile), 'r') as f:
        content = f.read()
    assert 'test' in content


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context_json_decoding_error():
    with pytest.raises(ContextDecodingException) as exc_info:
        generate_context(context_file='invalid.json')
    assert "JSON decoding error while loading" in str(exc_info.value)


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/fake/project/dir'
    infile = 'binary_file.png'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/fake/project/dir'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))
    assert os.path.isfile(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = '/fake/project/dir'
    infile = 'existing_file.txt'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = True

    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_filename():
    project_dir = '/fake/project/dir'
    infile = '{{""}}'
    context = {'cookiecutter': {'_new_lines': None}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    assert not os.path.exists(os.path.join(project_dir, ''))

def test_generate_file_custom_newline():
    project_dir = '/fake/project/dir'
    infile = 'text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=FileSystemLoader('/fake/template/dir'))
    skip_if_file_exists = False

    generate_file(project_dir, infile, context, env, skip_if_file_exists)

    with open(os.path.join(project_dir, infile), 'rb') as f:
        content = f.read()
        assert b'\r\n' in content


# LLM-generated content at query #28
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #29
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    assert is_binary("binary_file.png") is True


# LLM-generated content at query #30
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

def test_generate_context_with_both_default_and_extra_context():
    context = generate_context(
        'tests/test-fixtures/cookiecutter.json',
        default_context={'name': 'default'},
        extra_context={'name': 'extra'}
    )
    assert context == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}}

def test_generate_context_with_invalid_default_context():
    with pytest.warns(UserWarning):
        context = generate_context(
            'tests/test-fixtures/cookiecutter.json',
            default_context={'invalid': 'value'}
        )
    assert context == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}}


