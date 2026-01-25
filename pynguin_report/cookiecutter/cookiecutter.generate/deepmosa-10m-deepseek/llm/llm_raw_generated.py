####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__run_hook_from_repo_dir_deprecation_warning():
    repo_dir = '/path/to/repo'
    hook_name = 'test_hook'
    project_dir = '/path/to/project'
    context = {'key': 'value'}
    delete_project_on_failure = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "_run_hook_from_repo_dir" in str(w[-1].message)
        assert "run_hook_from_repo_dir" in str(w[-1].message)


def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    repo_dir = '/path/to/repo'
    hook_name = 'test_hook'
    project_dir = '/path/to/project'
    context = {'key': 'value'}
    delete_project_on_failure = True

    with unittest.mock.patch('cookiecutter.hooks.run_hook_from_repo_dir') as mock_run_hook:
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        mock_run_hook.assert_called_once_with(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    context = {'name': 'test'}
    output_dir = '/tmp'
    environment = Environment()
    dirname = '{{ name }}_dir'
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path('/tmp/test_dir')
    assert result[1] is True

def test_render_and_create_dir_raises_empty_dir_name_exception():
    context = {'name': 'test'}
    output_dir = '/tmp'
    environment = Environment()
    dirname = ''
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except EmptyDirNameException:
        pass
    else:
        assert False, 'Expected EmptyDirNameException'

def test_render_and_create_dir_raises_output_dir_exists_exception():
    context = {'name': 'test'}
    output_dir = '/tmp'
    environment = Environment()
    dirname = '{{ name }}_dir'
    Path('/tmp/test_dir').mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        pass
    else:
        assert False, 'Expected OutputDirExistsException'

def test_render_and_create_dir_overwrites_existing_directory():
    context = {'name': 'test'}
    output_dir = '/tmp'
    environment = Environment()
    dirname = '{{ name }}_dir'
    Path('/tmp/test_dir').mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == Path('/tmp/test_dir')
    assert result[1] is False


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {}


def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"nested": {}}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {}, "new_var": "value"}


def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a"]}


def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "a"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_apply_overwrites_to_context_choice_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_choice_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"a": 1, "b": 3}}


def test_apply_overwrites_to_context_bool_overwrite_valid():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new"}


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_files_with_valid_inputs():
    repo_dir = "path/to/repo"
    context = {"cookiecutter": {"project_name": "MyProject"}}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)


def test_generate_files_with_empty_context():
    repo_dir = "path/to/repo"
    context = {}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)


def test_generate_files_with_skip_if_file_exists():
    repo_dir = "path/to/repo"
    context = {"cookiecutter": {"project_name": "MyProject"}}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)


def test_generate_files_with_overwrite_if_exists():
    repo_dir = "path/to/repo"
    context = {"cookiecutter": {"project_name": "MyProject"}}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)


def test_generate_files_with_keep_project_on_failure():
    repo_dir = "path/to/repo"
    context = {"cookiecutter": {"project_name": "MyProject"}}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(project_dir)


def test_generate_files_with_copy_only_path():
    repo_dir = "path/to/repo"
    context = {"cookiecutter": {"project_name": "MyProject", "_copy_without_render": ["path/to/copy"]}}
    output_dir = "path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {}

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"dict_var": {"key1": "value1"}}
    overwrite_context = {"dict_var": {"key2": "value2"}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"dict_var": {"key1": "value1", "key2": "value2"}}

def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"list_var": ["choice1", "choice2"]}
    overwrite_context = {"list_var": "choice2"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"list_var": ["choice2", "choice1"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"list_var": ["choice1", "choice2"]}
    overwrite_context = {"list_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"multichoice_var": ["choice1", "choice2", "choice3"]}
    overwrite_context = {"multichoice_var": ["choice2", "choice3"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoice_var": ["choice2", "choice3"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"multichoice_var": ["choice1", "choice2"]}
    overwrite_context = {"multichoice_var": ["choice3"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_bool_valid_yes():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": True}

def test_apply_overwrites_to_context_bool_valid_no():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "no"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"bool_var": False}

def test_apply_overwrites_to_context_bool_invalid():
    context = {"bool_var": True}
    overwrite_context = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"simple_var": "old_value"}
    overwrite_context = {"simple_var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"simple_var": "new_value"}

def test_apply_overwrites_to_context_nested_dict():
    context = {"nested": {"level1": {"level2": "value"}}}
    overwrite_context = {"nested": {"level1": {"level2": "new_value"}}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"level1": {"level2": "new_value"}}}


# LLM-generated content at query #6
#--------------------------

```python
def test_YesNoPrompt_process_response_with_invalid_input():
    prompt = YesNoPrompt()
    invalid_input = "invalid_value"
    try:
        prompt.process_response(invalid_input)
    except InvalidResponse:
        pass
    else:
        assert False, "Expected InvalidResponse to be raised"


# LLM-generated content at query #7
#--------------------------

```python
def test_apply_overwrites_to_context_with_non_dict_overwrite():
    context = {"key": {"nested_key": "value"}}
    overwrite_context = {"key": "non_dict_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["key"] == "non_dict_value"


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_context_with_valid_json():
    context = generate_context(context_file='test.json')
    assert isinstance(context, OrderedDict)

def test_generate_context_with_invalid_json():
    try:
        generate_context(context_file='invalid.json')
    except ContextDecodingException:
        pass
    else:
        assert False

def test_generate_context_with_default_context():
    context = generate_context(context_file='test.json', default_context={'key': 'value'})
    assert 'key' in context['cookiecutter']

def test_generate_context_with_extra_context():
    context = generate_context(context_file='test.json', extra_context={'key': 'value'})
    assert 'key' in context['cookiecutter']

def test_generate_context_with_invalid_default_context():
    try:
        generate_context(context_file='test.json', default_context={'invalid_key': 'value'})
    except ValueError:
        pass
    else:
        assert False

def test_generate_context_with_invalid_extra_context():
    try:
        generate_context(context_file='test.json', extra_context={'invalid_key': 'value'})
    except ValueError:
        pass
    else:
        assert False


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory(tmp_path):
    context = {'name': 'test_project'}
    environment = Environment()
    dir_to_create, created = render_and_create_dir('{{ name }}', context, tmp_path, environment)
    assert dir_to_create.exists()
    assert created

def test_render_and_create_dir_raises_error_on_empty_dirname():
    context = {'name': 'test_project'}
    environment = Environment()
    try:
        render_and_create_dir('', context, '/tmp', environment)
    except EmptyDirNameException:
        assert True
    else:
        assert False

def test_render_and_create_dir_raises_error_on_existing_directory(tmp_path):
    context = {'name': 'test_project'}
    environment = Environment()
    dir_to_create = tmp_path / 'test_project'
    dir_to_create.mkdir()
    try:
        render_and_create_dir('{{ name }}', context, tmp_path, environment)
    except OutputDirExistsException:
        assert True
    else:
        assert False

def test_render_and_create_dir_overwrites_existing_directory(tmp_path):
    context = {'name': 'test_project'}
    environment = Environment()
    dir_to_create = tmp_path / 'test_project'
    dir_to_create.mkdir()
    dir_to_create, created = render_and_create_dir('{{ name }}', context, tmp_path, environment, overwrite_if_exists=True)
    assert dir_to_create.exists()
    assert not created


# LLM-generated content at query #10
#--------------------------

```python
def test_process_response_returns_true_for_yes_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("1") == True
    assert prompt.process_response("true") == True
    assert prompt.process_response("t") == True
    assert prompt.process_response("yes") == True
    assert prompt.process_response("y") == True
    assert prompt.process_response("on") == True


# LLM-generated content at query #11
#--------------------------

```python
def test_generate_files_creates_project_directory():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)

def test_generate_files_overwrites_existing_directory():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_skips_existing_files():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_accepts_hooks():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)

def test_generate_files_keeps_project_on_failure():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(project_dir)

def test_generate_files_raises_exception_on_empty_dirname():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": ""}}
    output_dir = "test_output"
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except EmptyDirNameException:
        assert True

def test_generate_files_raises_exception_on_existing_dir():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "test_output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except OutputDirExistsException:
        assert True


# LLM-generated content at query #12
#--------------------------

```python
def test_delete_project_on_failure_evaluates_to_true():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_path.exists()
    assert created

def test_render_and_create_dir_raises_empty_dir_name_exception():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_raises_output_dir_exists_exception():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        assert True

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_path.exists()
    assert not created


# LLM-generated content at query #14
#--------------------------

```
def test_is_copy_only_path_matches_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.txt', '*.json']}}
    assert is_copy_only_path('file.txt', context) == True

def test_is_copy_only_path_does_not_match_pattern():
    context = {'cookiecutter': {'_copy_without_render': ['*.txt', '*.json']}}
    assert is_copy_only_path('file.py', context) == False

def test_is_copy_only_path_no_copy_without_render_key():
    context = {'cookiecutter': {}}
    assert is_copy_only_path('file.txt', context) == False

def test_is_copy_only_path_empty_copy_without_render_list():
    context = {'cookiecutter': {'_copy_without_render': []}}
    assert is_copy_only_path('file.txt', context) == False


# LLM-generated content at query #15
#--------------------------

```python
def test_empty_dirname_raises_exception():
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    pytest.raises(EmptyDirNameException, render_and_create_dir, "", context, output_dir, environment)


# LLM-generated content at query #16
#--------------------------

```
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError to be raised"
    except ValueError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    repo_dir = "test_repo"
    context = {}
    output_dir = "output"
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = True

    generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)


# LLM-generated content at query #18
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {}
    overwrite_context = {'new_var': 'value'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {}

def test_apply_overwrites_to_context_new_variable_in_dictionary():
    context = {'dict_var': {}}
    overwrite_context = {'new_var': 'value'}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {'dict_var': {'new_var': 'value'}}

def test_apply_overwrites_to_context_multichoice_valid():
    context = {'multichoice_var': ['a', 'b', 'c']}
    overwrite_context = {'multichoice_var': ['a', 'b']}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'multichoice_var': ['a', 'b']}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {'multichoice_var': ['a', 'b', 'c']}
    overwrite_context = {'multichoice_var': ['a', 'd']}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable multichoice_var, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_choice_valid():
    context = {'choice_var': ['a', 'b', 'c']}
    overwrite_context = {'choice_var': 'b'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'choice_var': ['b', 'a', 'c']}

def test_apply_overwrites_to_context_choice_invalid():
    context = {'choice_var': ['a', 'b', 'c']}
    overwrite_context = {'choice_var': 'd'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice_var, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {'dict_var': {'key1': 'value1', 'key2': 'value2'}}
    overwrite_context = {'dict_var': {'key2': 'new_value2'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'dict_var': {'key1': 'value1', 'key2': 'new_value2'}}

def test_apply_overwrites_to_context_boolean_valid():
    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'yes'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'bool_var': True}

def test_apply_overwrites_to_context_boolean_invalid():
    context = {'bool_var': False}
    overwrite_context = {'bool_var': 'maybe'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "maybe provided for variable bool_var could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {'var': 'old_value'}
    overwrite_context = {'var': 'new_value'}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {'var': 'new_value'}


# LLM-generated content at query #19
#--------------------------

def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_choice provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #20
#--------------------------

```python
def test_process_response_with_invalid_choice():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid_choice")
    except InvalidResponse:
        pass
    else:
        assert False, "Expected InvalidResponse to be raised"


# LLM-generated content at query #21
#--------------------------

```python
def test_accept_hooks_evaluates_to_false():
    result = generate_files(repo_dir="test_repo", accept_hooks=False)
    assert "project_dir" in result


# LLM-generated content at query #22
#--------------------------

```python
def test_process_response_yes_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("1") == True
    assert prompt.process_response("true") == True
    assert prompt.process_response("t") == True
    assert prompt.process_response("yes") == True
    assert prompt.process_response("y") == True
    assert prompt.process_response("on") == True

def test_process_response_no_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("0") == False
    assert prompt.process_response("false") == False
    assert prompt.process_response("f") == False
    assert prompt.process_response("no") == False
    assert prompt.process_response("n") == False
    assert prompt.process_response("off") == False

def test_process_response_invalid_choice():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid")
    except InvalidResponse:
        pass
    else:
        assert False, "Expected InvalidResponse exception"


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_context_with_invalid_json():
    context_file = 'invalid.json'
    try:
        generate_context(context_file)
    except ContextDecodingException:
        pass


# LLM-generated content at query #24
#--------------------------

def test_generate_context_with_valid_json_file():
    import tempfile
    import os
    import json
    from collections import OrderedDict

    test_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        f.flush()
        context = generate_context(context_file=f.name)
        os.unlink(f.name)
    
    assert isinstance(context, OrderedDict)
    assert "cookiecutter" in context
    assert context["cookiecutter"] == test_data


# LLM-generated content at query #25
#--------------------------

def test_generate_context_with_valid_json():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"key": "value"}')
        f.flush()
        context = generate_context(context_file=f.name)
        assert context[os.path.splitext(os.path.basename(f.name))[0]] == {"key": "value"}
    os.unlink(f.name)

def test_generate_context_with_invalid_json():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('invalid json')
        f.flush()
        try:
            generate_context(context_file=f.name)
            assert False, "Should raise ContextDecodingException"
        except ContextDecodingException:
            pass
    os.unlink(f.name)

def test_generate_context_with_default_context():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"key": "original"}')
        f.flush()
        context = generate_context(context_file=f.name, default_context={"key": "default"})
        assert context[os.path.splitext(os.path.basename(f.name))[0]] == {"key": "default"}
    os.unlink(f.name)

def test_generate_context_with_extra_context():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"key": "original"}')
        f.flush()
        context = generate_context(context_file=f.name, extra_context={"key": "extra"})
        assert context[os.path.splitext(os.path.basename(f.name))[0]] == {"key": "extra"}
    os.unlink(f.name)

def test_generate_context_with_both_default_and_extra_context():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"key": "original"}')
        f.flush()
        context = generate_context(
            context_file=f.name,
            default_context={"key": "default"},
            extra_context={"key": "extra"}
        )
        assert context[os.path.splitext(os.path.basename(f.name))[0]] == {"key": "extra"}
    os.unlink(f.name)

def test_generate_context_with_nested_dictionaries():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"nested": {"key": "original"}}')
        f.flush()
        context = generate_context(
            context_file=f.name,
            extra_context={"nested": {"key": "new"}}
        )
        assert context[os.path.splitext(os.path.basename(f.name))[0]] == {"nested": {"key": "new"}}
    os.unlink(f.name)


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_context_with_valid_json():
    import tempfile
    import json
    import os

    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name)
        os.unlink(f.name)
    assert result == {"cookiecutter": context_data}

def test_generate_context_with_default_context():
    import tempfile
    import json
    import os

    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    default_context = {"key1": "new_value"}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name, default_context=default_context)
        os.unlink(f.name)
    assert result["cookiecutter"]["key1"] == "new_value"

def test_generate_context_with_extra_context():
    import tempfile
    import json
    import os

    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    extra_context = {"key1": "extra_value"}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result["cookiecutter"]["key1"] == "extra_value"

def test_generate_context_with_invalid_json():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        f.write("invalid json")
        f.flush()
        try:
            generate_context(f.name)
            assert False, "Should raise ContextDecodingException"
        except ContextDecodingException:
            pass
        os.unlink(f.name)

def test_generate_context_with_boolean_overwrite():
    import tempfile
    import json
    import os

    context_data = {"key1": True}
    extra_context = {"key1": "yes"}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result["cookiecutter"]["key1"] is True

def test_generate_context_with_list_overwrite():
    import tempfile
    import json
    import os

    context_data = {"key1": ["a", "b", "c"]}
    extra_context = {"key1": ["a", "b"]}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result["cookiecutter"]["key1"] == ["a", "b"]

def test_generate_context_with_dict_overwrite():
    import tempfile
    import json
    import os

    context_data = {"key1": {"subkey1": "value1", "subkey2": "value2"}}
    extra_context = {"key1": {"subkey1": "new_value"}}
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result["cookiecutter"]["key1"]["subkey1"] == "new_value"
    assert result["cookiecutter"]["key1"]["subkey2"] == "value2"


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_files_with_existing_output_dir_and_overwrite():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/path/to/output'
    overwrite_if_exists = True
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)


def test_generate_files_with_existing_output_dir_and_no_overwrite():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = False

    try:
        generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)
    except OutputDirExistsException:
        pass


def test_generate_files_with_skip_if_file_exists():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = True
    accept_hooks = True
    keep_project_on_failure = False

    generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)


def test_generate_files_without_hooks():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = False
    keep_project_on_failure = False

    generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)


def test_generate_files_with_keep_project_on_failure():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/path/to/output'
    overwrite_if_exists = False
    skip_if_file_exists = False
    accept_hooks = True
    keep_project_on_failure = True

    generate_files(repo_dir, context, output_dir, overwrite_if_exists, skip_if_file_exists, accept_hooks, keep_project_on_failure)


# LLM-generated content at query #28
#--------------------------

```python
def test_delete_project_on_failure_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False

def test_delete_project_on_failure_false_when_keep_project_on_failure_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False

def test_delete_project_on_failure_false_when_both_conditions_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #29
#--------------------------

def test_generate_context_with_valid_json():
    import tempfile
    import json
    import os
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        json.dump({'key': 'value'}, tmp)
        tmp.flush()
        result = generate_context(tmp.name)
        expected = OrderedDict([(tmp.name.split('.')[0], {'key': 'value'})])
        assert result == expected
    os.unlink(tmp.name)

def test_generate_context_with_invalid_json():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        tmp.write('invalid json')
        tmp.flush()
        try:
            generate_context(tmp.name)
            assert False, "Should raise ContextDecodingException"
        except ContextDecodingException:
            pass
    os.unlink(tmp.name)

def test_generate_context_with_default_context():
    import tempfile
    import json
    import os
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        json.dump({'key': ['value1', 'value2']}, tmp)
        tmp.flush()
        result = generate_context(tmp.name, default_context={'key': 'value1'})
        expected = OrderedDict([(tmp.name.split('.')[0], {'key': ['value1', 'value2']})])
        assert result == expected
    os.unlink(tmp.name)

def test_generate_context_with_extra_context():
    import tempfile
    import json
    import os
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        json.dump({'key': 'value'}, tmp)
        tmp.flush()
        result = generate_context(tmp.name, extra_context={'key': 'new_value'})
        expected = OrderedDict([(tmp.name.split('.')[0], {'key': 'new_value'})])
        assert result == expected
    os.unlink(tmp.name)

def test_generate_context_with_both_default_and_extra_context():
    import tempfile
    import json
    import os
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        json.dump({'key': ['value1', 'value2']}, tmp)
        tmp.flush()
        result = generate_context(
            tmp.name,
            default_context={'key': 'value1'},
            extra_context={'key': 'value2'}
        )
        expected = OrderedDict([(tmp.name.split('.')[0], {'key': ['value2', 'value1']})])
        assert result == expected
    os.unlink(tmp.name)

def test_generate_context_with_boolean_conversion():
    import tempfile
    import json
    import os
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        json.dump({'flag': True}, tmp)
        tmp.flush()
        result = generate_context(tmp.name, extra_context={'flag': 'yes'})
        expected = OrderedDict([(tmp.name.split('.')[0], {'flag': True})])
        assert result == expected
    os.unlink(tmp.name)


# LLM-generated content at query #30
#--------------------------

```
def test_generate_files_raises_undefined_error_when_directory_creation_fails():
    repo_dir = "test_repo"
    context = {"invalid_key": "value"}
    output_dir = "output_dir"
    overwrite_if_exists = True
    try:
        generate_files(repo_dir, context, output_dir, overwrite_if_exists)
        assert False, "Expected UndefinedError to be raised"
    except UndefinedError:
        pass


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_files_with_valid_input():
    repo_dir = "/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)

def test_generate_files_with_overwrite_if_exists():
    repo_dir = "/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    os.makedirs(output_dir)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_with_skip_if_file_exists():
    repo_dir = "/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_without_accept_hooks():
    repo_dir = "/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(project_dir)

def test_generate_files_with_keep_project_on_failure():
    repo_dir = "/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    assert os.path.exists(project_dir)

def test_generate_files_with_invalid_repo_dir():
    repo_dir = "/invalid/path/to/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/path/to/output"
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except Exception:
        assert True

def test_generate_files_with_empty_context():
    repo_dir = "/path/to/repo"
    context = {}
    output_dir = "/path/to/output"
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except Exception:
        assert True


# LLM-generated content at query #32
#--------------------------

def test_delete_project_on_failure_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False

def test_delete_project_on_failure_false_when_keep_project_on_failure_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_context_with_non_existent_file():
    try:
        generate_context(context_file="non_existent_file.json")
    except FileNotFoundError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_62_evaluates_to_true():
    repo_dir = Path("test_repo")
    context = {"cookiecutter": {"_jinja2_env_vars": {}}}
    env = create_env_with_context(context)
    template_dir = find_template(repo_dir, env)
    with work_in(template_dir):
        result = any(os.walk('.'))
    assert result


# LLM-generated content at query #35
#--------------------------

```python
def test_generate_context_with_empty_default_context():
    context = generate_context(
        context_file='test.json',
        default_context={},
        extra_context=None
    )
    assert 'test' in context


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_context_with_valid_json():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        context = generate_context(context_file=f.name)
    assert context == {'cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json")
        try:
            generate_context(context_file=f.name)
        except ContextDecodingException:
            pass
        else:
            assert False, "Expected ContextDecodingException"

def test_generate_context_with_default_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        context = generate_context(context_file=f.name, default_context={"key": "new_value"})
    assert context == {'cookiecutter': {'key': 'new_value'}}

def test_generate_context_with_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        context = generate_context(context_file=f.name, extra_context={"key": "new_value"})
    assert context == {'cookiecutter': {'key': 'new_value'}}

def test_generate_context_with_invalid_default_context():
    import json
    import tempfile
    import warnings
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        with warnings.catch_warnings(record=True) as w:
            generate_context(context_file=f.name, default_context={"invalid_key": "value"})
            assert len(w) == 1
            assert "Invalid default received" in str(w[0].message)

def test_generate_context_with_both_default_and_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        context = generate_context(
            context_file=f.name,
            default_context={"key": "default_value"},
            extra_context={"key": "extra_value"}
        )
    assert context == {'cookiecutter': {'key': 'extra_value'}}


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_false():
    context = {'cookiecutter': {'_jinja2_env_vars': {}}}
    env = create_env_with_context(context)
    repo_dir = '/path/to/repo'
    template_dir = find_template(repo_dir, env)
    project_dir = '/path/to/project'
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure == False


# LLM-generated content at query #38
#--------------------------

```python
def test_undefined_error_raises_undefined_variable_in_template():
    repo_dir = Path("test_repo")
    context = {"cookiecutter": {"invalid_var": "{{ undefined_var }}"}}
    output_dir = Path("output")
    try:
        generate_files(repo_dir, context, output_dir)
        assert False, "Expected UndefinedVariableInTemplate to be raised"
    except UndefinedVariableInTemplate:
        assert True


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"binary content")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"binary content"

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "r") as f:
        assert f.read() == "Hello World"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)

def test_generate_file_new_lines_config():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    context["name"] = "World"
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "r") as f:
        assert f.read() == "Hello World"


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_file_creates_file_with_correct_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "expected rendered content"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, "template.txt"), "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary_file.bin"), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_uses_custom_newline():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value", "_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", newline="") as f:
        content = f.read()
    assert "\r\n" in content

def test_generate_file_preserves_file_permissions():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("content")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert os.stat(os.path.join(project_dir, "template.txt")).st_mode & 0o777 == 0o644


# LLM-generated content at query #41
#--------------------------

```python
def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {}
    env = Environment()
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"binary content")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "binary.bin")
    with open(outfile, "rb") as f:
        assert f.read() == b"binary content"

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "r") as f:
        assert f.read() == "Hello value"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, infile), exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, infile))

def test_generate_file_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello\nWorld")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "r") as f:
        assert f.read() == "Hello\nWorld"

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ invalid_syntax }")
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError"
    except TemplateSyntaxError:
        pass


# LLM-generated content at query #42
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/test_project"
    infile = "test_binary.bin"
    context = {}
    env = Environment()
    
    # Create a dummy binary file
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = "/tmp/test_project"
    infile = "test_template.txt"
    context = {"name": "Test"}
    env = Environment(loader=DictLoader({infile: "Hello {{ name }}"}))
    
    # Create a dummy template file
    with open(infile, 'w') as f:
        f.write("Hello {{ name }}")
    
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile) as f:
        assert f.read() == "Hello Test"

def test_generate_file_skip_existing():
    project_dir = "/tmp/test_project"
    infile = "test_skip.txt"
    context = {}
    env = Environment()
    
    # Create dummy files
    with open(infile, 'w') as f:
        f.write("content")
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write("existing content")
    
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == "existing content"

def test_generate_file_empty_filename():
    project_dir = "/tmp/test_project"
    infile = ""
    context = {}
    env = Environment()
    
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_newlines():
    project_dir = "/tmp/test_project"
    infile = "test_newlines.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({infile: "line1\nline2"}))
    
    with open(infile, 'w') as f:
        f.write("line1\nline2")
    
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'rb') as f:
        assert b'\r\n' in f.read()


# LLM-generated content at query #43
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/binary_file"
    context = {}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/text_file"
    context = {"cookiecutter": {"key": "value"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.key }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r") as f:
        assert f.read() == "Hello value"

def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/skip_file"
    context = {}
    env = Environment()
    with open(infile, "w") as f:
        f.write("content")
    outfile = os.path.join(project_dir, infile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/text_file"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Line1\nLine2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r", newline="") as f:
        assert f.read() == "Line1\r\nLine2"


# LLM-generated content at query #44
#--------------------------

```python
def test_new_lines_from_context():
    context = {
        'cookiecutter': {
            '_new_lines': '\n'
        }
    }
    assert context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_False():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert "new" not in context

def test_apply_overwrites_to_context_new_dictionary_variable_added():
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["nested"]["new"] == "value"

def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["choices"] == ["b", "a"]

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d", "a"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert "d" in str(e)

def test_apply_overwrites_to_context_choice_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["choice"][0] == "b"

def test_apply_overwrites_to_context_choice_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert "d" in str(e)

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["nested"]["b"] == 3

def test_apply_overwrites_to_context_bool_overwrite_valid():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["flag"] is True

def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert "invalid" in str(e)

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite_context = {"var": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context["var"] == "new"


# LLM-generated content at query #2
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context(context_file='tests/test_data/valid_context.json')
    assert isinstance(context, dict)
    assert 'valid_context' in context

def test_generate_context_with_invalid_json_file():
    try:
        generate_context(context_file='tests/test_data/invalid_context.json')
        assert False, "Expected ContextDecodingException"
    except ContextDecodingException:
        pass

def test_generate_context_with_default_context():
    default_context = {'key1': 'value1'}
    context = generate_context(context_file='tests/test_data/valid_context.json', default_context=default_context)
    assert isinstance(context, dict)
    assert 'valid_context' in context

def test_generate_context_with_extra_context():
    extra_context = {'key2': 'value2'}
    context = generate_context(context_file='tests/test_data/valid_context.json', extra_context=extra_context)
    assert isinstance(context, dict)
    assert 'valid_context' in context

def test_generate_context_with_both_default_and_extra_context():
    default_context = {'key1': 'value1'}
    extra_context = {'key2': 'value2'}
    context = generate_context(context_file='tests/test_data/valid_context.json', default_context=default_context, extra_context=extra_context)
    assert isinstance(context, dict)
    assert 'valid_context' in context


# LLM-generated content at query #3
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    result_dir, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result_dir.exists()
    assert created

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    dirname = ""
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_raises_exception_for_existing_directory(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    existing_dir = output_dir / dirname
    existing_dir.mkdir()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

def test_render_and_create_dir_overwrites_existing_directory(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    existing_dir = output_dir / dirname
    existing_dir.mkdir()
    result_dir, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result_dir.exists()
    assert not created


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    dirname = "{{ name }}_dir"
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_path == Path("/tmp/test_dir")
    assert created is True
    assert dir_path.exists()

def test_render_and_create_dir_raises_error_when_dirname_is_empty():
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    dirname = ""
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_raises_error_when_directory_exists():
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    dirname = "{{ name }}_dir"
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

def test_render_and_create_dir_overwrites_existing_directory():
    context = {"name": "test"}
    output_dir = Path("/tmp")
    environment = Environment()
    dirname = "{{ name }}_dir"
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_path == Path("/tmp/test_dir")
    assert created is False
    assert dir_path.exists()


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_overwrites_to_context_new_first_level_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_deep_level_variable():
    context = {"nested": {"existing": "value"}}
    overwrite_context = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new": "value"}}

def test_apply_overwrites_to_context_overwrite_list():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["b", "a"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a"]}

def test_apply_overwrites_to_context_overwrite_list_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError as e:
        assert str(e) == "['d'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_overwrite_choice():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_overwrite_choice_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_overwrite_dict_partial():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"a": 1, "b": 3}}

def test_apply_overwrites_to_context_overwrite_bool_valid():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_overwrite_bool_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_overwrite_simple_value():
    context = {"key": "old"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new"}


# LLM-generated content at query #6
#--------------------------

```
def test_generate_file_creates_output_file():
    import tempfile
    import os
    import shutil
    from jinja2 import Environment, FileSystemLoader

    # Setup test environment
    project_dir = tempfile.mkdtemp()
    template_dir = tempfile.mkdtemp()
    env = Environment(loader=FileSystemLoader(template_dir))
    context = {'cookiecutter': {}}

    # Create test input file
    infile = os.path.join(template_dir, 'test.txt')
    with open(infile, 'w') as f:
        f.write('Hello {{ name }}')
    
    context['cookiecutter']['name'] = 'World'

    # Call function
    generate_file(project_dir, 'test.txt', context, env)

    # Verify output
    outfile = os.path.join(project_dir, 'test.txt')
    assert os.path.exists(outfile)
    with open(outfile, 'r') as f:
        assert f.read() == 'Hello World'

    # Cleanup
    shutil.rmtree(project_dir)
    shutil.rmtree(template_dir)

def test_generate_file_skips_existing_file():
    import tempfile
    import os
    import shutil
    from jinja2 import Environment, FileSystemLoader

    # Setup test environment
    project_dir = tempfile.mkdtemp()
    template_dir = tempfile.mkdtemp()
    env = Environment(loader=FileSystemLoader(template_dir))
    context = {'cookiecutter': {}}

    # Create test input file and existing output file
    infile = os.path.join(template_dir, 'test.txt')
    with open(infile, 'w') as f:
        f.write('Hello {{ name }}')
    
    outfile = os.path.join(project_dir, 'test.txt')
    with open(outfile, 'w') as f:
        f.write('Existing content')

    context['cookiecutter']['name'] = 'World'

    # Call function with skip_if_file_exists=True
    generate_file(project_dir, 'test.txt', context, env, skip_if_file_exists=True)

    # Verify output wasn't changed
    with open(outfile, 'r') as f:
        assert f.read() == 'Existing content'

    # Cleanup
    shutil.rmtree(project_dir)
    shutil.rmtree(template_dir)

def test_generate_file_copies_binary_file():
    import tempfile
    import os
    import shutil
    from jinja2 import Environment, FileSystemLoader

    # Setup test environment
    project_dir = tempfile.mkdtemp()
    template_dir = tempfile.mkdtemp()
    env = Environment(loader=FileSystemLoader(template_dir))
    context = {'cookiecutter': {}}

    # Create test binary file
    infile = os.path.join(template_dir, 'test.bin')
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')

    # Call function
    generate_file(project_dir, 'test.bin', context, env)

    # Verify output
    outfile = os.path.join(project_dir, 'test.bin')
    assert os.path.exists(outfile)
    with open(outfile, 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

    # Cleanup
    shutil.rmtree(project_dir)
    shutil.rmtree(template_dir)

def test_generate_file_handles_empty_filename():
    import tempfile
    import os
    import shutil
    from jinja2 import Environment, FileSystemLoader

    # Setup test environment
    project_dir = tempfile.mkdtemp()
    template_dir = tempfile.mkdtemp()
    env = Environment(loader=FileSystemLoader(template_dir))
    context = {'cookiecutter': {'name': ''}}

    # Create test input file that renders to empty string
    infile = os.path.join(template_dir, '{{ name }}.txt')
    with open(infile, 'w') as f:
        f.write('content')

    # Call function
    generate_file(project_dir, '{{ name }}.txt', context, env)

    # Verify no file was created
    outfile = os.path.join(project_dir, '.txt')
    assert not os.path.exists(outfile)

    # Cleanup
    shutil.rmtree(project_dir)
    shutil.rmtree(template_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "tests/data/binary_file.bin"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "tests/data/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "tests/data/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "tests/data/empty_file_name.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "tests/data/bad_template.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True


# LLM-generated content at query #8
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = '/path/to/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    skip_if_file_exists = True
    os.path.exists = lambda path: True
    generate_file(project_dir, infile, context, env, skip_if_file_exists)


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_context_with_valid_json():
    context_file = "tests/test_data/valid_context.json"
    default_context = {"key1": "value1"}
    extra_context = {"key2": "value2"}
    context = generate_context(context_file, default_context, extra_context)
    assert "valid_context" in context
    assert context["valid_context"]["key1"] == "value1"
    assert context["valid_context"]["key2"] == "value2"

def test_generate_context_with_invalid_json():
    context_file = "tests/test_data/invalid_context.json"
    try:
        generate_context(context_file)
    except ContextDecodingException:
        assert True
    else:
        assert False

def test_generate_context_with_no_default_and_extra_context():
    context_file = "tests/test_data/valid_context.json"
    context = generate_context(context_file)
    assert "valid_context" in context

def test_generate_context_with_invalid_default_context():
    context_file = "tests/test_data/valid_context.json"
    default_context = {"key1": "invalid_value"}
    try:
        generate_context(context_file, default_context)
    except ValueError:
        assert True
    else:
        assert False

def test_generate_context_with_invalid_extra_context():
    context_file = "tests/test_data/valid_context.json"
    extra_context = {"key2": "invalid_value"}
    try:
        generate_context(context_file, extra_context=extra_context)
    except ValueError:
        assert True
    else:
        assert False


# LLM-generated content at query #10
#--------------------------

```python
def test_apply_overwrites_to_context_new_first_level_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_deep_level_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": "value"}

def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["a", "b"]}

def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_choice_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_choice_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"a": 1, "b": 3}}

def test_apply_overwrites_to_context_bool_overwrite_valid():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old_value"}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new_value"}


# LLM-generated content at query #11
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/example/project"
    infile = "example.txt"
    context = {"key": "value"}
    env = Environment()
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("test content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert os.path.exists(outfile)


# LLM-generated content at query #12
#--------------------------

```python
def test_render_and_create_dir_overwrite_if_exists():
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dirname = "test_dir"
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create.exists()
    assert not created


# LLM-generated content at query #13
#--------------------------

```
def test_is_binary_predicate_evaluates_to_true():
    import tempfile
    import os
    from cookiecutter.utils import is_binary

    # Create a temporary binary file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        tmp_file.write(b'\x00\x01\x02\x03')
        tmp_file_path = tmp_file.name

    assert is_binary(tmp_file_path) is True
    os.unlink(tmp_file_path)


# LLM-generated content at query #14
#--------------------------

def test_generate_context_with_invalid_json_file():
    invalid_json_file = "tests/fixtures/invalid.json"
    try:
        generate_context(context_file=invalid_json_file)
    except ContextDecodingException as e:
        assert "JSON decoding error while loading" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_template_syntax_error_raised_when_invalid_template():
    project_dir = "/tmp/project"
    infile = "invalid_template.txt"
    context = {"cookiecutter": {"key": "value"}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        pass
    else:
        assert False, "Expected TemplateSyntaxError to be raised"


# LLM-generated content at query #16
#--------------------------

def test_generate_context_with_valid_json_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "test_project"}')
    context = generate_context(str(context_file))
    assert context == {"cookiecutter": {"name": "test_project"}}

def test_generate_context_with_invalid_json_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "test_project"')
    try:
        generate_context(str(context_file))
        assert False
    except ContextDecodingException:
        assert True

def test_generate_context_with_default_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "test_project"}')
    default_context = {"name": "new_project"}
    context = generate_context(str(context_file), default_context=default_context)
    assert context == {"cookiecutter": {"name": "new_project"}}

def test_generate_context_with_extra_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "test_project"}')
    extra_context = {"name": "extra_project"}
    context = generate_context(str(context_file), extra_context=extra_context)
    assert context == {"cookiecutter": {"name": "extra_project"}}

def test_generate_context_with_both_default_and_extra_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": "test_project"}')
    default_context = {"name": "default_project"}
    extra_context = {"name": "extra_project"}
    context = generate_context(str(context_file), default_context=default_context, extra_context=extra_context)
    assert context == {"cookiecutter": {"name": "extra_project"}}

def test_generate_context_with_invalid_default_context(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"name": ["test_project"]}')
    default_context = {"name": "invalid_project"}
    try:
        generate_context(str(context_file), default_context=default_context)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("test")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_context_with_valid_json():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": "value"}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name)
        assert context == {"cookiecutter": {"key": "value"}}

def test_generate_context_with_invalid_json():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_file.write("invalid json")
        tmp_file.flush()
        try:
            generate_context(context_file=tmp_file.name)
        except ContextDecodingException:
            pass

def test_generate_context_with_default_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": "value"}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name, default_context={"key": "new_value"})
        assert context == {"cookiecutter": {"key": "new_value"}}

def test_generate_context_with_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": "value"}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name, extra_context={"key": "new_value"})
        assert context == {"cookiecutter": {"key": "new_value"}}

def test_generate_context_with_default_and_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": "value"}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name, default_context={"key": "default_value"}, extra_context={"key": "new_value"})
        assert context == {"cookiecutter": {"key": "new_value"}}

def test_generate_context_with_multichoice_variable():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": ["value1", "value2"]}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name, extra_context={"key": ["value1", "value2"]})
        assert context == {"cookiecutter": {"key": ["value1", "value2"]}}

def test_generate_context_with_boolean_variable():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": True}, tmp_file)
        tmp_file.flush()
        context = generate_context(context_file=tmp_file.name, extra_context={"key": "yes"})
        assert context == {"cookiecutter": {"key": True}}

def test_generate_context_with_invalid_boolean_variable():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump({"key": True}, tmp_file)
        tmp_file.flush()
        try:
            generate_context(context_file=tmp_file.name, extra_context={"key": "invalid"})
        except ValueError:
            pass


# LLM-generated content at query #19
#--------------------------

```python
def test_is_copy_only_path_matching_pattern():
    path = "templates/example.txt"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*.txt"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_non_matching_pattern():
    path = "templates/example.doc"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*.txt"]}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_empty_context():
    path = "templates/example.txt"
    context = {}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_missing_copy_without_render():
    path = "templates/example.txt"
    context = {"cookiecutter": {}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_multiple_patterns():
    path = "templates/example.txt"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*.doc", "templates/*.txt"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_no_matching_patterns():
    path = "templates/example.doc"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*.txt", "templates/*.pdf"]}}
    assert is_copy_only_path(path, context) == False


# LLM-generated content at query #20
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError as e:
        assert e.translated == False


# LLM-generated content at query #21
#--------------------------

Here are the test cases:


# LLM-generated content at query #22
#--------------------------

def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)

def test__run_hook_from_repo_dir_emits_deprecation_warning():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    with warnings.catch_warnings(record=True) as w:
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "_run_hook_from_repo_dir' function is deprecated" in str(w[0].message)


# LLM-generated content at query #23
#--------------------------

def test_render_and_create_dir_successful_creation():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        env = Environment()
        context = {'name': 'test_project'}
        dirname = '{{ name }}'
        output_dir = temp_dir
        result_path, created = render_and_create_dir(dirname, context, output_dir, env)
        expected_path = temp_dir / 'test_project'
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
    finally:
        shutil.rmtree(temp_dir)


def test_render_and_create_dir_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from cookiecutter.exceptions import EmptyDirNameException

    env = Environment()
    context = {}
    dirname = ''
    output_dir = Path('/tmp')
    try:
        render_and_create_dir(dirname, context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


def test_render_and_create_dir_exists_no_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    from cookiecutter.exceptions import OutputDirExistsException
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        existing_dir = temp_dir / 'existing'
        existing_dir.mkdir()
        env = Environment()
        context = {}
        dirname = 'existing'
        output_dir = temp_dir
        try:
            render_and_create_dir(dirname, context, output_dir, env)
            assert False, "Expected OutputDirExistsException"
        except OutputDirExistsException:
            pass
    finally:
        shutil.rmtree(temp_dir)


def test_render_and_create_dir_exists_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        existing_dir = temp_dir / 'existing'
        existing_dir.mkdir()
        env = Environment()
        context = {}
        dirname = 'existing'
        output_dir = temp_dir
        result_path, created = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        assert result_path == existing_dir
        assert created is False
        assert existing_dir.exists()
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_new_lines_in_context():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False) == '\n'


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "Hello Test"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    with open(os.path.join(project_dir, "template.txt"), "w") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "Existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary.bin"), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_uses_newline_from_context():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test", "_new_lines": "\r\n"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", newline="") as f:
        content = f.read()
    assert content == "Hello Test\r\n"


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_files_with_copy_only_path():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'_copy_without_render': ['*.txt']}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'file.txt'), 'w') as f:
        f.write('test')
    generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'file.txt'))

def test_generate_files_with_rendered_path():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'name': 'project'}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'file.txt'), 'w') as f:
        f.write('{{ cookiecutter.name }}')
    generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'file.txt'))
    with open(os.path.join(output_dir, 'file.txt'), 'r') as f:
        assert f.read() == 'project'

def test_generate_files_with_overwrite_if_exists():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'name': 'project'}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'file.txt'), 'w') as f:
        f.write('{{ cookiecutter.name }}')
    with open(os.path.join(output_dir, 'file.txt'), 'w') as f:
        f.write('old content')
    generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'file.txt'))
    with open(os.path.join(output_dir, 'file.txt'), 'r') as f:
        assert f.read() == 'project'

def test_generate_files_with_skip_if_file_exists():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'name': 'project'}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'file.txt'), 'w') as f:
        f.write('{{ cookiecutter.name }}')
    with open(os.path.join(output_dir, 'file.txt'), 'w') as f:
        f.write('old content')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(output_dir, 'file.txt'))
    with open(os.path.join(output_dir, 'file.txt'), 'r') as f:
        assert f.read() == 'old content'

def test_generate_files_with_hooks():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'name': 'project'}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'pre_gen_project.py'), 'w') as f:
        f.write('print("pre_gen_project")')
    with open(os.path.join(repo_dir, 'post_gen_project.py'), 'w') as f:
        f.write('print("post_gen_project")')
    generate_files(repo_dir, context, output_dir, accept_hooks=True)

def test_generate_files_with_keep_project_on_failure():
    repo_dir = '/tmp/repo'
    context = {'cookiecutter': {'name': 'project'}}
    output_dir = '/tmp/output'
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(repo_dir, 'file.txt'), 'w') as f:
        f.write('{{ invalid_template }}')
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except UndefinedError:
        pass
    assert os.path.exists(output_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_render_and_create_dir_overwrite_if_exists():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = True
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
    assert result == (dir_to_create, False)


# LLM-generated content at query #28
#--------------------------

```python
def test_new_lines_in_context():
    context = {'cookiecutter': {'_new_lines': True}}
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #29
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname_raises_exception():
    environment = Environment()
    try:
        render_and_create_dir("", {}, "output_dir", environment)
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException"

def test_render_and_create_dir_with_existing_dir_and_no_overwrite_raises_exception():
    environment = Environment()
    context = {"name": "test"}
    output_dir = "output_dir"
    dirname = "test_dir"
    Path(output_dir).mkdir(exist_ok=True)
    Path(output_dir, dirname).mkdir(exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=False)
    except OutputDirExistsException:
        pass
    else:
        assert False, "Expected OutputDirExistsException"

def test_render_and_create_dir_with_existing_dir_and_overwrite_does_not_raise_exception():
    environment = Environment()
    context = {"name": "test"}
    output_dir = "output_dir"
    dirname = "test_dir"
    Path(output_dir).mkdir(exist_ok=True)
    Path(output_dir, dirname).mkdir(exist_ok=True)
    render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert Path(output_dir, dirname).exists()

def test_render_and_create_dir_with_non_existing_dir_creates_dir():
    environment = Environment()
    context = {"name": "test"}
    output_dir = "output_dir"
    dirname = "test_dir"
    Path(output_dir).mkdir(exist_ok=True)
    render_and_create_dir(dirname, context, output_dir, environment)
    assert Path(output_dir, dirname).exists()

def test_render_and_create_dir_returns_correct_tuple():
    environment = Environment()
    context = {"name": "test"}
    output_dir = "output_dir"
    dirname = "test_dir"
    Path(output_dir).mkdir(exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result == (Path(output_dir, dirname), True)


# LLM-generated content at query #30
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    try:
        render_and_create_dir("", {}, "output_dir", Environment())
    except EmptyDirNameException as e:
        assert str(e) == 'Error: directory name is empty'


# LLM-generated content at query #31
#--------------------------

```python
def test_generate_file_binary_copy():
    project_dir = "/tmp/project"
    infile = "tests/data/binary_file.bin"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    assert filecmp.cmp(infile, os.path.join(project_dir, infile), shallow=False)

def test_generate_file_text_render():
    project_dir = "/tmp/project"
    infile = "tests/data/text_file.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))
    with open(infile, "r", encoding="utf-8") as f1, open(os.path.join(project_dir, infile), "r", encoding="utf-8") as f2:
        assert f1.read() != f2.read()

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "tests/data/text_file.txt"
    context = {}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r", encoding="utf-8") as f:
        assert f.read() == "existing content"

def test_generate_file_empty_filename():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "tests/data/invalid_template.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError"
    except TemplateSyntaxError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_template_syntax_error_raised_when_invalid_template():
    project_dir = "/tmp/project"
    infile = "invalid_template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    exception_raised = False
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        exception_raised = True
    assert exception_raised == True


# LLM-generated content at query #33
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file"
    context = {}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "binary_file"))
    with open(os.path.join(project_dir, "binary_file"), "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "text_file.txt"))
    with open(os.path.join(project_dir, "text_file.txt"), "r") as f:
        assert f.read() == "Hello test"

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello")
    outfile = os.path.join(project_dir, "text_file.txt")
    with open(outfile, "w") as f:
        f.write("Existing")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "Existing"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/empty_file"
    context = {"cookiecutter": {"name": ""}}
    env = Environment()
    os.makedirs(os.path.join(project_dir, ""))
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt"), "r", newline="") as f:
        assert f.read() == "Hello\r\n"


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file"
    context = {}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b"binary data")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "binary_file"))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "text_file.txt"))

def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/skip_file.txt"
    context = {}
    env = Environment()
    with open(infile, "w") as f:
        f.write("content")
    outfile = os.path.join(project_dir, "skip_file.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "/tmp/template/invalid_template.txt"
    context = {}
    env = Environment()
    with open(infile, "w") as f:
        f.write("Hello {{ name")
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True
    else:
        assert False


# LLM-generated content at query #35
#--------------------------

def test_yes_no_prompt_invalid_response_raises_value_error():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_choice provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file.bin"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "binary_file.bin"))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "text_file.txt"))

def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, "text_file.txt"))

def test_generate_file_with_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "text_file.txt"))

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/empty_file"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "/tmp/template/invalid_template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True
    else:
        assert False


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_files_with_valid_input():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    output_dir = 'output_dir'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)

def test_generate_files_with_overwrite_if_exists():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    output_dir = 'output_dir'
    os.makedirs(os.path.join(output_dir, 'TestProject'))
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_with_skip_if_file_exists():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    output_dir = 'output_dir'
    os.makedirs(os.path.join(output_dir, 'TestProject'))
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_without_hooks():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    output_dir = 'output_dir'
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(project_dir)

def test_generate_files_with_keep_project_on_failure():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'TestProject'}}
    output_dir = 'output_dir'
    os.makedirs(os.path.join(output_dir, 'TestProject'))
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except Exception:
        assert os.path.exists(os.path.join(output_dir, 'TestProject'))


