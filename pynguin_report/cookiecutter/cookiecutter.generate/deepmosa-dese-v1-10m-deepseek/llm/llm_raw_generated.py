####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'test'}}
    delete_project_on_failure = True
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    try:
        render_and_create_dir('', {}, Path('/tmp'), Environment(), False)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_with_existing_dir_no_overwrite():
    existing_dir = Path('/tmp/existing_dir')
    existing_dir.mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir('existing_dir', {}, Path('/tmp'), Environment(), False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

def test_render_and_create_dir_with_existing_dir_and_overwrite():
    existing_dir = Path('/tmp/existing_dir')
    existing_dir.mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir('existing_dir', {}, Path('/tmp'), Environment(), True)
    assert result[0] == existing_dir
    assert result[1] == False

def test_render_and_create_dir_with_non_existing_dir():
    non_existing_dir = Path('/tmp/non_existing_dir')
    if non_existing_dir.exists():
        non_existing_dir.rmdir()
    result = render_and_create_dir('non_existing_dir', {}, Path('/tmp'), Environment(), False)
    assert result[0] == non_existing_dir
    assert result[1] == True
    assert non_existing_dir.exists()


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {}

def test_apply_overwrites_to_context_new_variable_in_dictionary():
    context = {"existing_var": {}}
    overwrite_context = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing_var": {}, "new_var": "value"}

def test_apply_overwrites_to_context_multichoice_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["a", "b"]}

def test_apply_overwrites_to_context_multichoice_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_choice_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite_context = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choice": ["b", "a", "c"]}

def test_apply_overwrites_to_context_choice_invalid():
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

def test_apply_overwrites_to_context_boolean_valid():
    context = {"flag": False}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_boolean_invalid():
    context = {"flag": False}
    overwrite_context = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "maybe provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old_value"}
    overwrite_context = {"var": "new_value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"var": "new_value"}


# LLM-generated content at query #4
#--------------------------

def test_apply_overwrites_to_context_choice_variable_invalid():
    context = {"choice_var": ["a", "b", "c"]}
    overwrite_context = {"choice_var": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice_var, but the choices are ['a', 'b', 'c']."


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_file_skips_existing_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.dat"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_uses_configured_newline():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_detects_newline():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #6
#--------------------------

```python
def test_skip_if_file_exists_and_file_already_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {}
    env = Environment()
    outfile = os.path.join(project_dir, "file.txt")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)


# LLM-generated content at query #7
#--------------------------

```python
def test_file_name_is_empty_evaluates_to_true():
    project_dir = "/tmp/project"
    infile = "template/file.txt"
    context = {"variable": "value"}
    env = Environment()
    outfile = os.path.join(project_dir, "")
    os.makedirs(outfile)
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #8
#--------------------------

def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_new_dict_variable_added():
    context = {"nested": {"existing": "value"}}
    overwrite = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new": "value"}}


def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choices": ["a", "b"]}


def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["x"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_choice_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_choice_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "x"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"nested": {"a": 1, "b": 3}}


def test_apply_overwrites_to_context_bool_valid_yes():
    context = {"flag": False}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_bool_valid_no():
    context = {"flag": True}
    overwrite = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_bool_invalid():
    context = {"flag": True}
    overwrite = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old"}
    overwrite = {"key": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"key": "new"}


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_files_creates_project_directory():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)

def test_generate_files_overwrites_existing_directory():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    os.makedirs(output_dir, exist_ok=True)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_skips_existing_files():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_runs_pre_and_post_hooks():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)

def test_generate_files_keeps_project_on_failure():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    try:
        project_dir = generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except Exception:
        assert os.path.exists(output_dir)

def test_generate_files_handles_copy_only_paths():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project', '_copy_without_render': ['*.txt']}}
    output_dir = '/tmp/output_dir'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)

def test_generate_files_raises_error_on_empty_directory_name():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': ''}}
    output_dir = '/tmp/output_dir'
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except EmptyDirNameException:
        assert True

def test_generate_files_raises_error_on_existing_directory_without_overwrite():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    os.makedirs(output_dir, exist_ok=True)
    try:
        generate_files(repo_dir, context, output_dir, overwrite_if_exists=False)
        assert False
    except OutputDirExistsException:
        assert True

def test_generate_files_handles_undefined_variables():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {}}  # Missing required 'project_name'
    output_dir = '/tmp/output_dir'
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except UndefinedVariableInTemplate:
        assert True

def test_generate_files_handles_template_syntax_errors():
    repo_dir = '/tmp/repo_dir'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output_dir'
    # Create a template file with syntax error
    with open(os.path.join(repo_dir, 'template.txt'), 'w') as f:
        f.write('{{ invalid_template }}')
    try:
        generate_files(repo_dir, context, output_dir)
        assert False
    except TemplateSyntaxError:
        assert True


# LLM-generated content at query #10
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    import jinja2
    env = jinja2.Environment()
    try:
        env.get_template("invalid_template")
    except jinja2.TemplateSyntaxError:
        assert True
    else:
        assert False


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path("/tmp/test_dir")
    assert result[1] == True

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == Path("/tmp/test_dir")
    assert result[1] == False

def test_render_and_create_dir_raises_exception_on_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except OutputDirExistsException:
        assert True

def test_render_and_create_dir_raises_exception_on_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #12
#--------------------------

```python
def test_apply_overwrites_to_context_with_invalid_boolean_overwrite():
    context = {'flag': True}
    overwrite_context = {'flag': 'invalid'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #13
#--------------------------

```python
def test_new_lines_configuration_evaluates_to_true():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False) != False


# LLM-generated content at query #14
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    import pytest

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "output_dir", None)


# LLM-generated content at query #15
#--------------------------

```
def test_template_syntax_error_raises_with_verbose_info():
    env = Environment()
    try:
        env.get_template("invalid_template.jinja")
    except TemplateSyntaxError as e:
        assert e.translated is False


# LLM-generated content at query #16
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    context = {}
    output_dir = Path("output")
    environment = Environment()
    
    try:
        render_and_create_dir("", context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = '/tmp/project'
    infile = 'test.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'w') as f:
        f.write('existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == 'existing content'

def test_generate_file_handles_binary_file():
    project_dir = '/tmp/project'
    infile = 'test.bin'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

def test_generate_file_renders_text_file():
    project_dir = '/tmp/project'
    infile = 'test.txt'
    context = {'cookiecutter': {'name': 'Test'}}
    env = Environment(loader=DictLoader({infile: 'Hello {{ cookiecutter.name }}'}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello {{ cookiecutter.name }}')
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile) as f:
        assert f.read() == 'Hello Test'

def test_generate_file_uses_custom_newline():
    project_dir = '/tmp/project'
    infile = 'test.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment(loader=DictLoader({infile: 'Hello\nWorld'}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello\nWorld')
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, 'rb') as f:
        assert f.read() == b'Hello\r\nWorld'

def test_generate_file_skips_empty_filename():
    project_dir = '/tmp/project'
    infile = ''
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ''))


# LLM-generated content at query #18
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/test/project"
    infile = "test.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("test")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)


# LLM-generated content at query #19
#--------------------------

def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_new_variable_in_dict():
    context = {"nested": {"existing": "value"}}
    overwrite = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new": "value"}}


def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choices": ["a", "b"]}


def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["x"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "['x'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"


def test_apply_overwrites_to_context_choice_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_choice_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "x"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "x provided for choice variable choice, but the choices are ['a', 'b', 'c']."


def test_apply_overwrites_to_context_nested_dict():
    context = {"nested": {"key": "value"}}
    overwrite = {"nested": {"key": "new_value"}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"nested": {"key": "new_value"}}


def test_apply_overwrites_to_context_bool_valid_yes():
    context = {"flag": False}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_bool_valid_no():
    context = {"flag": True}
    overwrite = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": False}


def test_apply_overwrites_to_context_bool_invalid():
    context = {"flag": True}
    overwrite = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "maybe provided for variable flag could not be converted to a boolean."


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old"}
    overwrite = {"key": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"key": "new"}


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_file_skips_existing_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    skip_if_file_exists = True
    # Pre-create the file to simulate existing file
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, "file.txt"), "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    # Assert no changes were made
    with open(os.path.join(project_dir, "file.txt"), "r") as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    # Create a binary file
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    # Assert the binary file was copied
    with open(os.path.join(project_dir, "file.bin"), "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment()
    # Create a text file with a template
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    # Assert the file was rendered
    with open(os.path.join(project_dir, "file.txt"), "r") as f:
        assert f.read() == "Hello Test"

def test_generate_file_uses_custom_newline():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    # Create a text file
    with open(infile, "w") as f:
        f.write("Line 1\nLine 2")
    generate_file(project_dir, infile, context, env)
    # Assert the file uses custom newline
    with open(os.path.join(project_dir, "file.txt"), "r") as f:
        assert f.read() == "Line 1\r\nLine 2"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"cookiecutter": {}}
    env = Environment()
    # Create a directory
    os.makedirs(infile, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    # Assert no file was created
    assert not os.path.isfile(os.path.join(project_dir, ""))


# LLM-generated content at query #21
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "output"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_with_empty_dirname_whitespace():
    dirname = "   "
    context = {}
    output_dir = "output"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skips_existing_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), "w").close()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"var": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_applies_file_permissions():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert os.stat(os.path.join(project_dir, infile)).st_mode & 0o777 == 0o644


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    skip_if_file_exists = True
    generate_file(project_dir, infile, context, env, skip_if_file_exists)

def test_generate_file_copies_binary_file():
    project_dir = "/path/to/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_renders_text_file():
    project_dir = "/path/to/project"
    infile = "text_file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_uses_configured_newline():
    project_dir = "/path/to/project"
    infile = "text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_handles_empty_file_name():
    project_dir = "/path/to/project"
    infile = ""
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #24
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"key": "value"}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    
    with unittest.mock.patch("os.path.exists", return_value=True):
        generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
        assert os.path.exists(outfile)


# LLM-generated content at query #25
#--------------------------

```python
def test_empty_dirname_raises_exception():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    import tempfile
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            render_and_create_dir("", {}, tmpdir, Environment())
        except EmptyDirNameException:
            pass
        else:
            assert False, "Expected EmptyDirNameException"

def test_whitespace_dirname_raises_exception():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    import tempfile
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            render_and_create_dir("   ", {}, tmpdir, Environment())
        except EmptyDirNameException:
            pass
        else:
            assert False, "Expected EmptyDirNameException"


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_47_evaluates_to_true():
    infile = "path/to/binary/file"
    assert is_binary(infile)


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"key": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.key }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "value"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"key": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp"))
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, infile), "w", encoding="utf-8") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, infile), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {"key": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"key": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp"))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, infile))

def test_generate_file_uses_custom_newline():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": "\r"}}
    env = Environment(loader=FileSystemLoader("/tmp"))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter._new_lines }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), "r", encoding="utf-8", newline="") as f:
        content = f.read()
    assert content == "\r"


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_context_with_valid_json():
    context_file = "cookiecutter.json"
    default_context = {"key1": "value1"}
    extra_context = {"key2": "value2"}
    context = generate_context(context_file, default_context, extra_context)
    assert context is not None

def test_generate_context_with_invalid_json():
    context_file = "invalid.json"
    default_context = {"key1": "value1"}
    extra_context = {"key2": "value2"}
    try:
        generate_context(context_file, default_context, extra_context)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

def test_generate_context_with_default_context():
    context_file = "cookiecutter.json"
    default_context = {"key1": "value1"}
    extra_context = None
    context = generate_context(context_file, default_context, extra_context)
    assert context is not None

def test_generate_context_with_extra_context():
    context_file = "cookiecutter.json"
    default_context = None
    extra_context = {"key2": "value2"}
    context = generate_context(context_file, default_context, extra_context)
    assert context is not None

def test_generate_context_with_no_additional_context():
    context_file = "cookiecutter.json"
    default_context = None
    extra_context = None
    context = generate_context(context_file, default_context, extra_context)
    assert context is not None


# LLM-generated content at query #29
#--------------------------

```
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {"existing": "value"}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dict_variable():
    context = {"existing": {"nested": "value"}}
    overwrite_context = {"new": "value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value", "new": "value"}}

def test_apply_overwrites_to_context_list_valid_overwrite():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": "b"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"choices": ["b", "a", "c"]}

def test_apply_overwrites_to_context_list_invalid_overwrite():
    context = {"choices": ["a", "b", "c"]}
    overwrite_context = {"choices": "d"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "d provided for choice variable choices, but the choices are ['a', 'b', 'c']."

def test_apply_overwrites_to_context_multichoice_valid_overwrite():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"multichoice": ["b", "c"]}

def test_apply_overwrites_to_context_multichoice_invalid_overwrite():
    context = {"multichoice": ["a", "b", "c"]}
    overwrite_context = {"multichoice": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "['b', 'd'] provided for multi-choice variable multichoice, but valid choices are ['a', 'b', 'c']"

def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite_context = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"nested": {"a": 1, "b": 3}}

def test_apply_overwrites_to_context_bool_valid_overwrite():
    context = {"flag": True}
    overwrite_context = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"flag": True}

def test_apply_overwrites_to_context_bool_invalid_overwrite():
    context = {"flag": True}
    overwrite_context = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "maybe provided for variable flag could not be converted to a boolean."

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old"}
    overwrite_context = {"key": "new"}
    apply_overwrites_to_context(context, overwrite_context)
    assert context == {"key": "new"}


# LLM-generated content at query #2
#--------------------------

```python
def test__run_hook_from_repo_dir():
    repo_dir = '/tmp/repo'
    hook_name = 'pre_gen_project'
    project_dir = '/tmp/project'
    context = {'cookiecutter': {'project_name': 'Test Project'}}
    delete_project_on_failure = True
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #3
#--------------------------

```python
def test_boolean_conversion_failure():
    context = {'test_var': True}
    overwrite_context = {'test_var': 'invalid_value'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_value provided for variable test_var could not be converted to a boolean


# LLM-generated content at query #4
#--------------------------

def test_render_and_create_dir_with_empty_dirname_raises_exception():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    try:
        render_and_create_dir('', context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_with_existing_dir_and_no_overwrite_raises_exception():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = 'existing_dir'
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment, False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass
    finally:
        dir_to_create.rmdir()

def test_render_and_create_dir_with_existing_dir_and_overwrite_returns_dir_and_false():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = 'existing_dir'
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, True)
    assert result == (dir_to_create, False)
    dir_to_create.rmdir()

def test_render_and_create_dir_with_non_existing_dir_creates_dir_and_returns_dir_and_true():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = 'new_dir'
    dir_to_create = Path(output_dir, dirname)
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result == (dir_to_create, True)
    assert dir_to_create.exists()
    dir_to_create.rmdir()

def test_render_and_create_dir_with_rendered_dirname_creates_correct_dir():
    context = {'name': 'project'}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = '{{ name }}_dir'
    dir_to_create = Path(output_dir, 'project_dir')
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result == (dir_to_create, True)
    assert dir_to_create.exists()
    dir_to_create.rmdir()


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context('tests/test-generate-context-valid.json')
    assert context == {'test': {'key1': 'value1', 'key2': 'value2'}}

def test_generate_context_with_invalid_json_file():
    try:
        generate_context('tests/test-generate-context-invalid.json')
        assert False
    except ContextDecodingException:
        assert True

def test_generate_context_with_default_context():
    context = generate_context(
        'tests/test-generate-context-valid.json',
        default_context={'key1': 'new_value1'}
    )
    assert context == {'test': {'key1': 'new_value1', 'key2': 'value2'}}

def test_generate_context_with_extra_context():
    context = generate_context(
        'tests/test-generate-context-valid.json',
        extra_context={'key1': 'new_value1'}
    )
    assert context == {'test': {'key1': 'new_value1', 'key2': 'value2'}}

def test_generate_context_with_default_and_extra_context():
    context = generate_context(
        'tests/test-generate-context-valid.json',
        default_context={'key1': 'default_value1'},
        extra_context={'key1': 'new_value1'}
    )
    assert context == {'test': {'key1': 'new_value1', 'key2': 'value2'}}

def test_generate_context_with_invalid_default_context():
    try:
        generate_context(
            'tests/test-generate-context-valid.json',
            default_context={'invalid_key': 'value'}
        )
        assert False
    except ValueError:
        assert True

def test_generate_context_with_invalid_extra_context():
    try:
        generate_context(
            'tests/test-generate-context-valid.json',
            extra_context={'invalid_key': 'value'}
        )
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_process_response_yes_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("1") is True
    assert prompt.process_response("true") is True
    assert prompt.process_response("t") is True
    assert prompt.process_response("yes") is True
    assert prompt.process_response("y") is True
    assert prompt.process_response("on") is True

def test_process_response_no_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("0") is False
    assert prompt.process_response("false") is False
    assert prompt.process_response("f") is False
    assert prompt.process_response("no") is False
    assert prompt.process_response("n") is False
    assert prompt.process_response("off") is False

def test_process_response_invalid_choice():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid")
        assert False
    except InvalidResponse:
        assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_handles_json_decoding_error():
    invalid_json_file = 'tests/fixtures/invalid.json'
    try:
        generate_context(context_file=invalid_json_file)
    except ContextDecodingException as e:
        assert str(e).startswith("JSON decoding error while loading")
        assert "Decoding error details" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_to_create == Path("/tmp/test_dir")
    assert created is True

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_raises_exception_for_existing_directory():
    dirname = "existing_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/existing_dir").mkdir(exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "existing_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/existing_dir").mkdir(exist_ok=True)
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create == Path("/tmp/existing_dir")
    assert created is False


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_files_creates_project_dir():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/fake/output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)


def test_generate_files_overwrites_existing_dir():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/fake/output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)


def test_generate_files_skips_existing_files():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/fake/output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    open(os.path.join(output_dir, "test_project", "existing_file.txt"), "w").close()
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, "existing_file.txt"))


def test_generate_files_runs_hooks():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "/fake/output"
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)


def test_generate_files_keeps_project_on_failure():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"invalid_var": "{{ undefined_var }}"}}
    output_dir = "/fake/output"
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except Exception:
        assert os.path.exists(os.path.join(output_dir, "{{ invalid_var }}"))


def test_generate_files_raises_error_on_invalid_template():
    repo_dir = "/fake/repo"
    context = {"cookiecutter": {"invalid_var": "{{ undefined_var }}"}}
    output_dir = "/fake/output"
    try:
        generate_files(repo_dir, context, output_dir)
        assert False, "Expected UndefinedVariableInTemplate to be raised"
    except UndefinedVariableInTemplate:
        assert True


# LLM-generated content at query #10
#--------------------------

def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_choice provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #11
#--------------------------

def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_new_dict_variable_added():
    context = {"nested": {}}
    overwrite = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"nested": {"new": "value"}}


def test_apply_overwrites_to_context_list_overwrite_valid_multichoice():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choices": ["a", "b"]}


def test_apply_overwrites_to_context_list_overwrite_invalid_multichoice():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "['a', 'd'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"


def test_apply_overwrites_to_context_single_choice_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_single_choice_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "d provided for choice variable choice, but the choices are ['a', 'b', 'c']."


def test_apply_overwrites_to_context_nested_dict_partial_overwrite():
    context = {"nested": {"key1": "value1", "key2": "value2"}}
    overwrite = {"nested": {"key1": "new_value"}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"nested": {"key1": "new_value", "key2": "value2"}}


def test_apply_overwrites_to_context_boolean_conversion_valid():
    context = {"flag": True}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_boolean_conversion_invalid():
    context = {"flag": True}
    overwrite = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"key": "old_value"}
    overwrite = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"key": "new_value"}


# LLM-generated content at query #12
#--------------------------

```python
def test_process_response_raises_invalid_response_for_invalid_input():
    prompt = YesNoPrompt()
    prompt.process_response("invalid_input")


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir_raises_exception_when_dirname_is_empty():
    try:
        render_and_create_dir(
            dirname="",
            context={},
            output_dir="test_output",
            environment=Environment(),
            overwrite_if_exists=False,
        )
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    overwrite_if_exists = True
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    rendered_dir, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
    
    assert rendered_dir == dir_to_create
    assert created is False
    assert dir_to_create.exists()


# LLM-generated content at query #15
#--------------------------

```python
def test_process_response_raises_invalid_response_when_value_is_invalid():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid")
    except InvalidResponse:
        pass
    else:
        assert False, "Expected InvalidResponse to be raised"


# LLM-generated content at query #16
#--------------------------

```python
def test_generate_file_handles_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"binary content")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary_file.bin"), "rb") as f:
        assert f.read() == b"binary content"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt"), "r", encoding="utf-8") as f:
        assert f.read() == "value"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/skip_file.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("content")
    outfile = os.path.join(project_dir, "skip_file.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r", encoding="utf-8") as f:
        assert f.read() == "existing content"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/empty_file"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "empty_file"))
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, "empty_file"))


# LLM-generated content at query #17
#--------------------------

```
def test_skip_if_file_exists_and_file_exists():
    import os
    import tempfile
    from jinja2 import Environment
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as project_dir:
        outfile = os.path.join(project_dir, "test.txt")
        with open(outfile, "w") as f:
            f.write("existing file")

        env = Environment()
        context = {}
        infile = "test.txt"

        with patch("os.path.exists", return_value=True):
            generate_file(project_dir, infile, context, env, skip_if_file_exists=True)


# LLM-generated content at query #18
#--------------------------

```
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)
    with open(outfile, "r") as f:
        assert f.read() == "existing content


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #20
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_path == Path("/tmp/test_dir")
    assert created

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_raises_exception_for_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except OutputDirExistsException:
        assert True

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    dir_path, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_path == Path("/tmp/test_dir")
    assert not created


# LLM-generated content at query #21
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = "/test/project"
    infile = "test_template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True
    else:
        assert False, "Expected TemplateSyntaxError to be raised"


# LLM-generated content at query #22
#--------------------------

def test_render_and_create_dir_overwrites_existing_dir_when_overwrite_flag_is_true():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.generate import render_and_create_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        existing_dir = output_dir / "existing_dir"
        existing_dir.mkdir()

        dirname = "existing_dir"
        context = {}
        environment = Environment()
        overwrite_if_exists = True

        result_path, created = render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists
        )

        assert result_path == existing_dir
        assert not created
        assert existing_dir.exists()


# LLM-generated content at query #23
#--------------------------

```python
def test_generate_file_binary():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.bin"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "file.bin"))
    with open(os.path.join(project_dir, "file.bin"), "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_text():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ key }}")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "file.txt"))
    with open(os.path.join(project_dir, "file.txt"), "r", encoding="utf-8") as f:
        assert f.read() == "Hello value"

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ key }}")
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r", encoding="utf-8") as f:
        assert f.read() == "Existing content"

def test_generate_file_empty_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(infile, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_newline_configuration():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}, "key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ key }}")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "file.txt"))
    with open(os.path.join(project_dir, "file.txt"), "r", encoding="utf-8") as f:
        assert f.read() == "Hello value\r\n"


# LLM-generated content at query #24
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "Hello value"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    with open(os.path.join(project_dir, "template.txt"), "w") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "template.txt"), "r") as f:
        content = f.read()
    assert content == "Existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary_file.bin"), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)

def test_generate_file_uses_configured_newline():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value", "_new_lines": "\r\n"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", newline="") as f:
        content = f.read()
    assert content == "Hello value\r\n"

def test_generate_file_applies_file_permissions():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert os.stat(os.path.join(project_dir, "template.txt")).st_mode & 0o777 == 0o644


# LLM-generated content at query #25
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #26
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

def test_generate_file_handles_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.dat"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_uses_configured_newlines():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_detects_newlines():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_skips_empty_filename():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #27
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_to_create == Path("/tmp/test_dir")
    assert created is True

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create == Path("/tmp/test_dir")
    assert created is False

def test_render_and_create_dir_raises_exception_for_existing_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    Path("/tmp/test_dir").mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        assert True


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #29
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    import pytest

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir(
            dirname="",
            context={},
            output_dir="/tmp",
            environment=None,
        )


# LLM-generated content at query #30
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    import os
    import tempfile
    import json
    from collections import OrderedDict

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json_content = {'key': 'value'}
        json.dump(json_content, tmp)
        tmp_path = tmp.name

    try:
        context = generate_context(context_file=tmp_path)
        assert isinstance(context, OrderedDict)
        assert tmp_path.split('/')[-1].split('.')[0] in context
        assert context[tmp_path.split('/')[-1].split('.')[0]] == json_content
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_template_syntax_error_raised():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    env.get_template = lambda _: exec("raise TemplateSyntaxError('error', 1, 'test', 'test')")
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError to be raised"
    except TemplateSyntaxError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_new_dict_variable_added():
    context = {"nested": {"existing": "value"}}
    overwrite = {"nested": {"new": "value"}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"nested": {"existing": "value", "new": "value"}}


def test_apply_overwrites_to_context_list_overwrite_valid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choices": ["a", "b"]}


def test_apply_overwrites_to_context_list_overwrite_invalid():
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["x"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_choice_overwrite_valid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"choice": ["b", "a", "c"]}


def test_apply_overwrites_to_context_choice_overwrite_invalid():
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "x"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_dict_partial_overwrite():
    context = {"nested": {"a": 1, "b": 2}}
    overwrite = {"nested": {"b": 3}}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"nested": {"a": 1, "b": 3}}


def test_apply_overwrites_to_context_bool_overwrite_valid():
    context = {"flag": False}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"flag": True}


def test_apply_overwrites_to_context_bool_overwrite_invalid():
    context = {"flag": False}
    overwrite = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"value": "old"}
    overwrite = {"value": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"value": "new"}


# LLM-generated content at query #2
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
    output_dir = '/tmp'
    try:
        render_and_create_dir(dirname, context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_already_exists_no_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil
    from cookiecutter.exceptions import OutputDirExistsException

    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / 'existing_dir').mkdir()
        env = Environment()
        context = {}
        dirname = 'existing_dir'
        output_dir = temp_dir
        try:
            render_and_create_dir(dirname, context, output_dir, env)
            assert False, "Expected OutputDirExistsException"
        except OutputDirExistsException:
            pass
    finally:
        shutil.rmtree(temp_dir)

def test_render_and_create_dir_already_exists_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / 'existing_dir').mkdir()
        env = Environment()
        context = {}
        dirname = 'existing_dir'
        output_dir = temp_dir
        result_path, created = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        expected_path = temp_dir / 'existing_dir'
        assert result_path == expected_path
        assert created is False
        assert expected_path.exists()
    finally:
        shutil.rmtree(temp_dir)

def test_render_and_create_dir_with_template_variables():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        env = Environment()
        context = {'project': 'awesome', 'version': '1.0'}
        dirname = '{{ project }}-v{{ version }}'
        output_dir = temp_dir
        result_path, created = render_and_create_dir(dirname, context, output_dir, env)
        expected_path = temp_dir / 'awesome-v1.0'
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_is_copy_only_path_matches_pattern():
    path = "templates/example.txt"
    context = {"cookiecutter": {"_copy_without_render": ["templates/*"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_does_not_match_pattern():
    path = "templates/example.txt"
    context = {"cookiecutter": {"_copy_without_render": ["other/*"]}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_no_copy_without_render_key():
    path = "templates/example.txt"
    context = {"cookiecutter": {}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_empty_copy_without_render_list():
    path = "templates/example.txt"
    context = {"cookiecutter": {"_copy_without_render": []}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_none_context():
    path = "templates/example.txt"
    context = None
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_empty_path():
    path = ""
    context = {"cookiecutter": {"_copy_without_render": ["templates/*"]}}
    assert is_copy_only_path(path, context) == False


# LLM-generated content at query #4
#--------------------------

```python
def test_generate_context_basic():
    context_file = 'cookiecutter.json'
    default_context = {'key1': 'value1'}
    extra_context = {'key2': 'value2'}
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key1": "default_value1", "key2": "default_value2"}')
    context = generate_context(context_file, default_context, extra_context)
    assert context['cookiecutter']['key1'] == 'value1'
    assert context['cookiecutter']['key2'] == 'value2'
    os.remove(context_file)

def test_generate_context_invalid_json():
    context_file = 'cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')
    try:
        generate_context(context_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"
    os.remove(context_file)

def test_generate_context_with_default_context_only():
    context_file = 'cookiecutter.json'
    default_context = {'key1': 'value1'}
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key1": "default_value1"}')
    context = generate_context(context_file, default_context)
    assert context['cookiecutter']['key1'] == 'value1'
    os.remove(context_file)

def test_generate_context_with_extra_context_only():
    context_file = 'cookiecutter.json'
    extra_context = {'key2': 'value2'}
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key2": "default_value2"}')
    context = generate_context(context_file, extra_context=extra_context)
    assert context['cookiecutter']['key2'] == 'value2'
    os.remove(context_file)

def test_generate_context_without_default_or_extra_context():
    context_file = 'cookiecutter.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('{"key1": "default_value1"}')
    context = generate_context(context_file)
    assert context['cookiecutter']['key1'] == 'default_value1'
    os.remove(context_file)


# LLM-generated content at query #5
#--------------------------

```python
def test_process_response_raises_InvalidResponse_for_invalid_input():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid")
    except InvalidResponse:
        pass
    else:
        assert False


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook_from_repo_dir_deprecation_warning():
    repo_dir = '/path/to/repo'
    hook_name = 'hook_name'
    project_dir = '/path/to/project'
    context = {'key': 'value'}
    delete_project_on_failure = True
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "_run_hook_from_repo_dir' function is deprecated" in str(w[0].message)

def test_run_hook_from_repo_dir_calls_run_hook_from_repo_dir():
    repo_dir = '/path/to/repo'
    hook_name = 'hook_name'
    project_dir = '/path/to/project'
    context = {'key': 'value'}
    delete_project_on_failure = True
    
    with unittest.mock.patch('cookiecutter.hooks.run_hook_from_repo_dir') as mock_run_hook:
        _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)
        
        mock_run_hook.assert_called_once_with(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Project Name: {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Project Name: Test Project"

def test_generate_file_skips_existing_file_with_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Project Name: {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Project Name: Test Project"
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Project Name: Test Project"

def test_generate_file_copies_binary_file_without_rendering():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary.bin"), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_uses_new_line_from_context():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test Project", "_new_lines": "\r\n"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Project Name: {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Project Name: Test Project"


# LLM-generated content at query #8
#--------------------------

```python
def test_file_name_is_empty_evaluates_to_true():
    project_dir = "/path/to/project"
    infile = "template/file.txt"
    context = {}
    env = None
    outfile_tmpl = env.from_string(infile)
    outfile = os.path.join(project_dir, outfile_tmpl.render(**context))
    os.makedirs(outfile)
    file_name_is_empty = os.path.isdir(outfile)
    assert file_name_is_empty == True


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir_raises_on_empty_dirname():
    try:
        render_and_create_dir(
            dirname="",
            context={},
            output_dir="some_dir",
            environment=Environment(),
        )
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_invalid_boolean_overwrite():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid provided for variable flag could not be converted to a boolean."


# LLM-generated content at query #11
#--------------------------

def test_render_and_create_dir_raises_on_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment

    try:
        render_and_create_dir("", {}, "output", Environment(), False)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_creates_new_dir():
    from pathlib import Path
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    import tempfile
    import shutil

    output_dir = Path(tempfile.mkdtemp())
    try:
        dir_path, created = render_and_create_dir("test", {}, output_dir, Environment(), False)
        assert created
        assert dir_path.exists()
        assert dir_path == output_dir / "test"
    finally:
        shutil.rmtree(output_dir)

def test_render_and_create_dir_raises_on_existing_dir():
    from pathlib import Path
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    import tempfile
    import shutil

    output_dir = Path(tempfile.mkdtemp())
    try:
        (output_dir / "test").mkdir()
        try:
            render_and_create_dir("test", {}, output_dir, Environment(), False)
            assert False, "Expected OutputDirExistsException"
        except OutputDirExistsException:
            pass
    finally:
        shutil.rmtree(output_dir)

def test_render_and_create_dir_overwrites_existing_dir():
    from pathlib import Path
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    import tempfile
    import shutil

    output_dir = Path(tempfile.mkdtemp())
    try:
        (output_dir / "test").mkdir()
        dir_path, created = render_and_create_dir("test", {}, output_dir, Environment(), True)
        assert not created
        assert dir_path.exists()
        assert dir_path == output_dir / "test"
    finally:
        shutil.rmtree(output_dir)

def test_render_and_create_dir_renders_template():
    from pathlib import Path
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    import tempfile
    import shutil

    output_dir = Path(tempfile.mkdtemp())
    try:
        context = {"name": "rendered"}
        dir_path, created = render_and_create_dir("{{ name }}", context, output_dir, Environment(), False)
        assert created
        assert dir_path.exists()
        assert dir_path == output_dir / "rendered"
    finally:
        shutil.rmtree(output_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/existing_file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/empty_file"
    context = {"cookiecutter": {}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_with_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #13
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {}
    env = Environment()
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, "template.txt")
    
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    
    with open(outfile, "r") as f:
        content = f.read()
    
    assert content == "existing content"


# LLM-generated content at query #14
#--------------------------

```python
def test_render_and_create_dir_existing_output_dir():
    dirname = "existing_dir"
    context = {}
    output_dir = Path("/tmp/test")
    environment = Environment()
    overwrite_if_exists = False

    # Create the directory to make it exist
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)

    # This should raise OutputDirExistsException since the directory exists
    try:
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
    except OutputDirExistsException:
        pass
    else:
        assert False, "Expected OutputDirExistsException to be raised"


# LLM-generated content at query #15
#--------------------------

```python
def test_template_syntax_error_has_translated_disabled():
    from jinja2 import Environment, TemplateSyntaxError
    env = Environment()
    try:
        env.get_template("invalid_template.jinja2")
    except TemplateSyntaxError as exception:
        assert exception.translated is False


# LLM-generated content at query #16
#--------------------------

```python
def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError not raised"
    except ValueError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_boolean_variable_with_invalid_response():
    context = {'test_var': True}
    overwrite_context = {'test_var': 'invalid'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError to be raised"
    except ValueError:
        assert True


# LLM-generated content at query #18
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = '/tmp/project'
    infile = '/tmp/binary_file'
    context = {'cookiecutter': {}}
    env = Environment()
    open(infile, 'wb').write(b'\x00\x01\x02')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text_file():
    project_dir = '/tmp/project'
    infile = '/tmp/text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    with open(infile, 'w') as f:
        f.write('Hello, World!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_skip_if_file_exists():
    project_dir = '/tmp/project'
    infile = '/tmp/text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    with open(infile, 'w') as f:
        f.write('Hello, World!')
    outfile = os.path.join(project_dir, infile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        f.write('Existing Content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, 'r') as f:
        assert f.read() == 'Existing Content'

def test_generate_file_empty_file_name():
    project_dir = '/tmp/project'
    infile = '/tmp/empty_file'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(infile)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_new_lines_config():
    project_dir = '/tmp/project'
    infile = '/tmp/text_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment()
    with open(infile, 'w') as f:
        f.write('Hello, World!')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'r', newline='') as f:
        assert f.newlines == '\r\n'


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_is_binary_predicate_evaluates_to_true():
    project_dir = "/path/to/project"
    infile = "/path/to/binary/file"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)


# LLM-generated content at query #21
#--------------------------

```python
def test_process_response_raises_invalid_response():
    prompt = YesNoPrompt()
    try:
        prompt.process_response("invalid")
    except InvalidResponse:
        pass
    else:
        assert False, "Expected InvalidResponse to be raised"


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_context_with_valid_context_file():
    context = generate_context(context_file='tests/test-context.json')
    assert isinstance(context, OrderedDict)
    assert 'test_context' in context

def test_generate_context_with_invalid_context_file():
    try:
        generate_context(context_file='tests/invalid-context.json')
        assert False, "Expected ContextDecodingException"
    except ContextDecodingException:
        pass

def test_generate_context_with_default_context():
    default_context = {'key1': 'value1'}
    context = generate_context(context_file='tests/test-context.json', default_context=default_context)
    assert context['test_context']['key1'] == 'value1'

def test_generate_context_with_extra_context():
    extra_context = {'key2': 'value2'}
    context = generate_context(context_file='tests/test-context.json', extra_context=extra_context)
    assert context['test_context']['key2'] == 'value2'

def test_generate_context_with_invalid_default_context():
    default_context = {'invalid_key': 'invalid_value'}
    try:
        generate_context(context_file='tests/test-context.json', default_context=default_context)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_generate_context_with_invalid_extra_context():
    extra_context = {'invalid_key': 'invalid_value'}
    try:
        generate_context(context_file='tests/test-context.json', extra_context=extra_context)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #23
#--------------------------

def test_render_and_create_dir_raises_on_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment
    import pytest

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir('', {}, '/tmp', Environment(), False)


def test_render_and_create_dir_creates_new_directory(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment

    output_dir = tmp_path / 'output'
    dir_path, created = render_and_create_dir('test', {}, output_dir, Environment())
    assert dir_path == output_dir / 'test'
    assert created
    assert dir_path.exists()


def test_render_and_create_dir_raises_on_existing_dir(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import OutputDirExistsException
    from jinja2 import Environment
    import pytest

    output_dir = tmp_path / 'output'
    (output_dir / 'test').mkdir(parents=True)
    with pytest.raises(OutputDirExistsException):
        render_and_create_dir('test', {}, output_dir, Environment(), False)


def test_render_and_create_dir_overwrites_existing_dir(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment

    output_dir = tmp_path / 'output'
    (output_dir / 'test').mkdir(parents=True)
    dir_path, created = render_and_create_dir('test', {}, output_dir, Environment(), True)
    assert dir_path == output_dir / 'test'
    assert not created
    assert dir_path.exists()


def test_render_and_create_dir_renders_template(tmp_path):
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment

    output_dir = tmp_path / 'output'
    context = {'name': 'rendered'}
    dir_path, created = render_and_create_dir('{{ name }}', context, output_dir, Environment())
    assert dir_path == output_dir / 'rendered'
    assert created
    assert dir_path.exists()


# LLM-generated content at query #24
#--------------------------

```python
def test_render_and_create_dir_existing_output_dir():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        pass  # Expected behavior
    else:
        assert False, "Expected OutputDirExistsException to be raised"


# LLM-generated content at query #25
#--------------------------

```python
def test_generate_file_template_syntax_error_handling():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError as exception:
        assert exception.translated is False


# LLM-generated content at query #26
#--------------------------

```python
def test_cookiecutter_new_lines_set_to_true():
    context = {'cookiecutter': {'_new_lines': True}}
    assert context['cookiecutter'].get('_new_lines', False) == True

def test_cookiecutter_new_lines_set_to_false():
    context = {'cookiecutter': {'_new_lines': False}}
    assert context['cookiecutter'].get('_new_lines', False) == False

def test_cookiecutter_new_lines_not_set():
    context = {'cookiecutter': {}}
    assert context['cookiecutter'].get('_new_lines', False) == False


# LLM-generated content at query #27
#--------------------------

```python
def test_render_and_create_dir_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    import pytest

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", {}, "output_dir", Environment())


# LLM-generated content at query #28
#--------------------------

```python
def test_generate_file_creates_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.bin"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"binary content")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "file.bin"), "rb") as f:
        assert f.read() == b"binary content"

def test_generate_file_creates_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello, {{ key }}!")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "file.txt"), "r") as f:
        assert f.read() == "Hello, value!"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello, {{ key }}!")
    existing_file = os.path.join(project_dir, "file.txt")
    with open(existing_file, "w") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(existing_file, "r") as f:
        assert f.read() == "Existing content"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"key": "value"}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(infile, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, ""))


# LLM-generated content at query #29
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/binary_file.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "binary_file.bin"))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/text_file.txt"
    context = {"name": "test"}
    env = Environment(loader=DictLoader({"text_file.txt": "Hello {{ name }}"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write("Hello {{ name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt")) as f:
        assert f.read() == "Hello test"

def test_generate_file_skip_if_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/text_file.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write("content")
    outfile = os.path.join(project_dir, "text_file.txt")
    with open(outfile, 'w') as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == "existing content"

def test_generate_file_empty_filename():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_newlines():
    project_dir = "/tmp/project"
    infile = "/tmp/text_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({"text_file.txt": "line1\nline2"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w', newline='\n') as f:
        f.write("line1\nline2")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt"), 'rb') as f:
        assert f.read() == b"line1\r\nline2"


# LLM-generated content at query #30
#--------------------------

def test_generate_files_creates_project_directory():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "output"
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert "test_project" in project_dir

def test_generate_files_handles_copy_only_paths():
    repo_dir = "test_repo"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "_copy_without_render": ["*.txt"]
        }
    }
    output_dir = "output"
    with open(os.path.join(repo_dir, "test.txt"), "w") as f:
        f.write("test")
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(project_dir, "test.txt"))

def test_generate_files_raises_on_existing_output_without_overwrite():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    try:
        generate_files(repo_dir, context, output_dir, overwrite_if_exists=False)
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass

def test_generate_files_overwrites_existing_output():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "output"
    os.makedirs(os.path.join(output_dir, "test_project"))
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)

def test_generate_files_runs_pre_and_post_hooks():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "output"
    os.makedirs(os.path.join(repo_dir, "hooks"))
    with open(os.path.join(repo_dir, "hooks", "pre_gen_project.py"), "w") as f:
        f.write("print('pre hook')")
    with open(os.path.join(repo_dir, "hooks", "post_gen_project.py"), "w") as f:
        f.write("print('post hook')")
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)

def test_generate_files_skips_hooks_when_disabled():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"project_name": "test_project"}}
    output_dir = "output"
    os.makedirs(os.path.join(repo_dir, "hooks"))
    with open(os.path.join(repo_dir, "hooks", "pre_gen_project.py"), "w") as f:
        f.write("raise Exception('This should not run')")
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert os.path.exists(project_dir)

def test_generate_files_keeps_project_on_failure():
    repo_dir = "test_repo"
    context = {"cookiecutter": {"invalid_var": "{{ invalid }}"}}
    output_dir = "output"
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
        assert False, "Should have raised UndefinedVariableInTemplate"
    except UndefinedVariableInTemplate:
        assert os.path.exists(os.path.join(output_dir, "{{ invalid }}"))


# LLM-generated content at query #31
#--------------------------

def test_render_and_create_dir_raises_empty_dir_name_exception_when_dirname_is_empty():
    empty_dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(empty_dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_raises_empty_dir_name_exception_when_dirname_is_whitespace():
    whitespace_dirname = "   "
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    try:
        render_and_create_dir(whitespace_dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #32
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    outfile = os.path.join(project_dir, "template.txt")
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        content = f.read()
    assert content == "existing content"


# LLM-generated content at query #33
#--------------------------

def test_render_and_create_dir_successful_creation():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        env = Environment()
        context = {'name': 'test_project'}
        dirname = "{{ name }}"
        output_dir = Path(temp_dir)
        result_path, created = render_and_create_dir(dirname, context, output_dir, env)
        assert result_path == Path(temp_dir) / "test_project"
        assert created is True
        assert result_path.exists()
    finally:
        shutil.rmtree(temp_dir)

def test_render_and_create_dir_empty_dirname():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from cookiecutter.exceptions import EmptyDirNameException

    env = Environment()
    context = {}
    dirname = ""
    output_dir = "some/path"
    try:
        render_and_create_dir(dirname, context, output_dir, env)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_already_exists_no_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    from cookiecutter.exceptions import OutputDirExistsException
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        existing_dir = Path(temp_dir) / "existing"
        existing_dir.mkdir()
        
        env = Environment()
        context = {}
        dirname = "existing"
        output_dir = Path(temp_dir)
        try:
            render_and_create_dir(dirname, context, output_dir, env)
            assert False, "Expected OutputDirExistsException"
        except OutputDirExistsException:
            pass
    finally:
        shutil.rmtree(temp_dir)

def test_render_and_create_dir_already_exists_with_overwrite():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        existing_dir = Path(temp_dir) / "existing"
        existing_dir.mkdir()
        
        env = Environment()
        context = {}
        dirname = "existing"
        output_dir = Path(temp_dir)
        result_path, created = render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=True)
        assert result_path == existing_dir
        assert created is False
        assert existing_dir.exists()
    finally:
        shutil.rmtree(temp_dir)

def test_render_and_create_dir_with_nested_path():
    from cookiecutter.generate import render_and_create_dir
    from jinja2 import Environment
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        env = Environment()
        context = {'name': 'test_project'}
        dirname = "parent/{{ name }}"
        output_dir = Path(temp_dir)
        result_path, created = render_and_create_dir(dirname, context, output_dir, env)
        assert result_path == Path(temp_dir) / "parent" / "test_project"
        assert created is True
        assert result_path.exists()
    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #34
#--------------------------

```python
def test_generate_file_binary():
    project_dir = "/tmp/test_project"
    infile = "test_binary.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = "/tmp/test_project"
    infile = "test_template.txt"
    context = {"name": "Test"}
    env = Environment(loader=DictLoader({infile: "Hello {{ name }}"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile, "r") as f:
        assert f.read() == "Hello Test"

def test_generate_file_skip_exists():
    project_dir = "/tmp/test_project"
    infile = "test_skip.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Test")
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "w") as f:
        f.write("Existing")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "Existing"

def test_generate_file_newlines():
    project_dir = "/tmp/test_project"
    infile = "test_newlines.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({infile: "Line1\nLine2"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Line1\nLine2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "r", newline="") as f:
        assert f.read() == "Line1\r\nLine2"


# LLM-generated content at query #35
#--------------------------

```python
def test_render_and_create_dir_creates_directory_when_not_exists():
    dirname = "test_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_to_create == Path("/tmp/test_dir")
    assert created is True

def test_render_and_create_dir_raises_exception_when_dirname_empty():
    dirname = ""
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except EmptyDirNameException:
        assert True
    else:
        assert False

def test_render_and_create_dir_raises_exception_when_dir_exists_and_no_overwrite():
    dirname = "existing_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        assert True
    else:
        assert False

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "existing_dir"
    context = {}
    output_dir = Path("/tmp")
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create == Path("/tmp/existing_dir")
    assert created is False


# LLM-generated content at query #36
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    from pathlib import Path
    from jinja2 import Environment
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = '{{ name }}_dir'
    result = render_and_create_dir(dirname, context, output_dir, environment)
    assert result[0] == Path('/tmp/test_dir')
    assert result[1] is True

def test_render_and_create_dir_raises_error_for_empty_dirname():
    from jinja2 import Environment
    context = {'name': 'test'}
    output_dir = '/tmp'
    environment = Environment()
    dirname = ''
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except EmptyDirNameException:
        assert True
    else:
        assert False

def test_render_and_create_dir_raises_error_for_existing_directory():
    from pathlib import Path
    from jinja2 import Environment
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = '{{ name }}_dir'
    Path('/tmp/test_dir').mkdir(parents=True, exist_ok=True)
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        assert True
    else:
        assert False

def test_render_and_create_dir_overwrites_existing_directory():
    from pathlib import Path
    from jinja2 import Environment
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    dirname = '{{ name }}_dir'
    Path('/tmp/test_dir').mkdir(parents=True, exist_ok=True)
    result = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result[0] == Path('/tmp/test_dir')
    assert result[1] is False


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_file_with_binary_file():
    project_dir = '/tmp/project'
    infile = '/tmp/template/binary_file'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, 'binary_file'))

def test_generate_file_with_text_file():
    project_dir = '/tmp/project'
    infile = '/tmp/template/text_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Hello, World!')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, 'text_file.txt'))

def test_generate_file_with_skip_if_file_exists():
    project_dir = '/tmp/project'
    infile = '/tmp/template/existing_file.txt'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, 'existing_file.txt'), 'w') as f:
        f.write('Existing content')
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, 'existing_file.txt'), 'r') as f:
        assert f.read() == 'Existing content'

def test_generate_file_with_new_lines():
    project_dir = '/tmp/project'
    infile = '/tmp/template/new_lines_file.txt'
    context = {'cookiecutter': {'_new_lines': '\r\n'}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, 'w') as f:
        f.write('Line 1\nLine 2')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, 'new_lines_file.txt'), 'r', newline='') as f:
        lines = f.readlines()
        assert lines[-1].endswith('\r\n')

def test_generate_file_with_empty_file_name():
    project_dir = '/tmp/project'
    infile = '/tmp/template/empty_file_name'
    context = {'cookiecutter': {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'empty_file_name'), exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, 'empty_file_name'))


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_file_handles_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file"
    context = {}
    env = Environment()
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary_file"), 'rb') as f:
        assert f.read() == b'\x00\x01\x02\x03'

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    with open(infile, 'w') as f:
        f.write("{{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt"), 'r') as f:
        assert f.read() == "value"

def test_generate_file_skips_existing_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/skip_file.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir)
    with open(os.path.join(project_dir, "skip_file.txt"), 'w') as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "skip_file.txt"), 'r') as f:
        assert f.read() == "existing content"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_preserves_file_permissions():
    project_dir = "/tmp/project"
    infile = "/tmp/template/permissions_file"
    context = {}
    env = Environment()
    with open(infile, 'w') as f:
        f.write("content")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert oct(os.stat(os.path.join(project_dir, "permissions_file")).st_mode & 0o777) == '0o644'

def test_generate_file_handles_newline_configuration():
    project_dir = "/tmp/project"
    infile = "/tmp/template/newline_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    with open(infile, 'w') as f:
        f.write("line1\nline2")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "newline_file.txt"), 'r') as f:
        assert f.read() == "line1\r\nline2"


# LLM-generated content at query #39
#--------------------------

```python
def test_skip_if_file_exists_predicate():
    project_dir = "/test/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    outfile_tmpl = env.from_string(infile)
    outfile = os.path.join(project_dir, outfile_tmpl.render(**context))
    
    # Create the file to ensure it exists
    with open(outfile, 'w') as f:
        f.write("test content")
    
    # Test that predicate evaluates to True when file exists and skip_if_file_exists is True
    assert skip_if_file_exists and os.path.exists(outfile)


# LLM-generated content at query #40
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/binary_file"
    context = {"key": "value"}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    assert is_binary(infile) == True


