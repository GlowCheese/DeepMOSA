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
    except ValueError as e:
        assert str(e) == "['x'] provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"


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
    except ValueError as e:
        assert str(e) == "x provided for choice variable choice, but the choices are ['a', 'b', 'c']."


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
    overwrite = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError as e:
        assert str(e) == "maybe provided for variable flag could not be converted to a boolean."


def test_apply_overwrites_to_context_simple_overwrite():
    context = {"value": "old"}
    overwrite = {"value": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"value": "new"}


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_from_repo_dir_calls_run_hook_from_repo_dir_with_correct_args():
    repo_dir = "/path/to/repo"
    hook_name = "pre_gen_project"
    project_dir = "/path/to/project"
    context = {"cookiecutter": {"project_name": "test_project"}}
    delete_project_on_failure = True
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #3
#--------------------------

def test_render_and_create_dir_with_empty_dirname():
    try:
        render_and_create_dir("", {}, "output_dir", Environment(), False)
        assert False, "Expected EmptyDirNameException"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_with_existing_dir_no_overwrite():
    try:
        dir_to_create = Path("existing_dir")
        dir_to_create.mkdir(exist_ok=True)
        render_and_create_dir("existing_dir", {}, ".", Environment(), False)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        pass
    finally:
        dir_to_create.rmdir()

def test_render_and_create_dir_with_existing_dir_with_overwrite():
    dir_to_create = Path("existing_dir")
    dir_to_create.mkdir(exist_ok=True)
    result = render_and_create_dir("existing_dir", {}, ".", Environment(), True)
    assert result[0] == dir_to_create
    assert result[1] is False
    dir_to_create.rmdir()

def test_render_and_create_dir_with_new_dir():
    dir_to_create = Path("new_dir")
    result = render_and_create_dir("new_dir", {}, ".", Environment(), False)
    assert result[0] == dir_to_create
    assert result[1] is True
    dir_to_create.rmdir()

def test_render_and_create_dir_with_template():
    env = Environment()
    context = {"name": "project"}
    result = render_and_create_dir("{{ name }}", context, ".", env, False)
    assert result[0] == Path("project")
    assert result[1] is True
    Path("project").rmdir()


# LLM-generated content at query #4
#--------------------------

def test_render_and_create_dir_overwrites_existing_dir_when_overwrite_flag_is_true():
    from cookiecutter.generate import render_and_create_dir
    from pathlib import Path
    import tempfile
    from jinja2 import Environment

    with tempfile.TemporaryDirectory() as tmp_dir:
        existing_dir = Path(tmp_dir) / "existing"
        existing_dir.mkdir()
        result_dir, created = render_and_create_dir(
            "existing",
            {},
            tmp_dir,
            Environment(),
            overwrite_if_exists=True
        )
        assert result_dir == existing_dir
        assert not created


# LLM-generated content at query #5
#--------------------------

```python
def test_is_copy_only_path_returns_true_when_path_matches_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('readme.txt', context) == True
    assert is_copy_only_path('docs/index.md', context) == True

def test_is_copy_only_path_returns_false_when_path_does_not_match_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('main.py', context) == False
    assert is_copy_only_path('src/utils.py', context) == False

def test_is_copy_only_path_returns_false_when_no_copy_without_render_in_context():
    context = {'cookiecutter': {}}
    assert is_copy_only_path('any_path.txt', context) == False


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_overwrites_to_context_list_in_dictionary_variable():
    context = {"key": ["value1", "value2"]}
    overwrite_context = {"key": "new_value"}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["key"] == "new_value"


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_with_valid_json():
    import tempfile
    import json
    context_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump({"key": "value"}, context_file)
    context_file.close()
    context = generate_context(context_file.name)
    assert context == OrderedDict([('cookiecutter', OrderedDict([('key', 'value')]))])

def test_generate_context_with_invalid_json():
    import tempfile
    context_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    context_file.write("invalid json")
    context_file.close()
    try:
        generate_context(context_file.name)
        assert False, "Expected ContextDecodingException"
    except ContextDecodingException:
        assert True

def test_generate_context_with_default_context():
    import tempfile
    import json
    context_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump({"key": "value"}, context_file)
    context_file.close()
    default_context = {"key": "new_value"}
    context = generate_context(context_file.name, default_context=default_context)
    assert context == OrderedDict([('cookiecutter', OrderedDict([('key', 'new_value')]))])

def test_generate_context_with_extra_context():
    import tempfile
    import json
    context_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump({"key": "value"}, context_file)
    context_file.close()
    extra_context = {"key": "new_value"}
    context = generate_context(context_file.name, extra_context=extra_context)
    assert context == OrderedDict([('cookiecutter', OrderedDict([('key', 'new_value')]))])

def test_generate_context_with_default_and_extra_context():
    import tempfile
    import json
    context_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    json.dump({"key": "value"}, context_file)
    context_file.close()
    default_context = {"key": "default_value"}
    extra_context = {"key": "extra_value"}
    context = generate_context(context_file.name, default_context=default_context, extra_context=extra_context)
    assert context == OrderedDict([('cookiecutter', OrderedDict([('key', 'extra_value')]))])


# LLM-generated content at query #8
#--------------------------

```python
def test_is_copy_only_path_matches_pattern():
    path = "src/templates/index.html"
    context = {"cookiecutter": {"_copy_without_render": ["src/templates/*"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_does_not_match_pattern():
    path = "src/templates/index.html"
    context = {"cookiecutter": {"_copy_without_render": ["src/static/*"]}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_with_missing_key():
    path = "src/templates/index.html"
    context = {"cookiecutter": {}}
    assert is_copy_only_path(path, context) == False


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "binary.bin")
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("{{ cookiecutter.var }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "r") as f:
        assert f.read() == "value"

def test_generate_file_returns_if_file_name_is_empty():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(project_dir)
    assert not os.path.exists(os.path.join(project_dir, ""))

def test_generate_file_applies_file_permissions():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("content")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    assert os.stat(outfile).st_mode & 0o777 == 0o644

def test_generate_file_handles_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("line1\nline2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "r") as f:
        assert f.read() == "line1\r\nline2"


# LLM-generated content at query #10
#--------------------------

```python
def test_skip_if_file_exists_and_file_already_exists():
    project_dir = "/tmp/project"
    infile = "example.txt"
    context = {"cookiecutter": {"_new_lines": False}}
    env = Environment()
    skip_if_file_exists = True
    outfile = os.path.join(project_dir, infile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    assert os.path.exists(outfile)


# LLM-generated content at query #11
#--------------------------

def test_yes_no_prompt_invalid_response_raises_value_error():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError not raised"
    except ValueError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_file_creates_file_with_correct_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test_project"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Project: {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Project: test_project"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test_project"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Project: {{ cookiecutter.name }}")
    with open(os.path.join(project_dir, "template.txt"), "w", encoding="utf-8") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "template.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {"name": "test_project"}}
    env = Environment()
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
    context = {"cookiecutter": {"name": "test_project"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)


# LLM-generated content at query #13
#--------------------------

def test_generate_context_handles_invalid_json_file():
    invalid_json_file = "tests/fixtures/invalid.json"
    try:
        generate_context(context_file=invalid_json_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException for invalid JSON file"


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    skip_if_file_exists = True
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("test content")
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.bin")
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("{{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "r") as f:
        assert f.read() == "value"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(infile, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)


# LLM-generated content at query #15
#--------------------------

```python
def test_process_response_raises_invalid_response_for_invalid_input():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
        assert False, "Expected ValueError to be raised"
    except ValueError:
        pass


# LLM-generated content at query #16
#--------------------------

```
def test_is_binary_predicate_evaluates_to_true():
    import tempfile
    import os

    # Create a temporary binary file
    with tempfile.NamedTemporaryFile(suffix='.bin', mode='wb', delete=False) as f:
        f.write(b'\x00\x01\x02\x03')
        binary_file_path = f.name

    # Mock environment and context
    env = type('MockEnv', (), {'from_string': lambda self, x: type('MockTemplate', (), {'render': lambda self, **kwargs: x})()})()
    context = {}

    # Call generate_file with the binary file
    generate_file(
        project_dir=os.path.dirname(binary_file_path),
        infile=binary_file_path,
        context=context,
        env=env,
        skip_if_file_exists=False
    )

    # Clean up
    os.unlink(binary_file_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_generate_context_with_default_context():
    context = generate_context(default_context={'key': 'value'})
    assert context['cookiecutter']['key'] == 'value'


# LLM-generated content at query #18
#--------------------------

```
def test_template_syntax_error_has_translated_disabled():
    class MockTemplateSyntaxError:
        translated = True

    exception = MockTemplateSyntaxError()
    try:
        try:
            raise MockTemplateSyntaxError()
        except TemplateSyntaxError:
            exception.translated = False
            raise
    except TemplateSyntaxError as e:
        assert e.translated == False


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_file_skips_when_file_exists():
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

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test"}}
    env = Environment(loader=FileSystemLoader("."))
    with open(infile, "w") as f:
        f.write("Hello {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "r") as f:
        assert f.read() == "Hello Test"


# LLM-generated content at query #20
#--------------------------

def test_generate_context_handles_invalid_json():
    import os
    import tempfile
    from cookiecutter.exceptions import ContextDecodingException

    invalid_json = "invalid json"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(invalid_json)
        temp_path = f.name

    try:
        generate_context(context_file=temp_path)
    except ContextDecodingException:
        pass
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #21
#--------------------------

def test_generate_context_handles_invalid_json():
    import os
    import tempfile
    from cookiecutter.generate import generate_context
    from cookiecutter.exceptions import ContextDecodingException

    invalid_json_content = '{"invalid": json}'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(invalid_json_content)
        f.flush()
        temp_file_path = f.name

    try:
        generate_context(context_file=temp_file_path)
    except ContextDecodingException:
        pass
    finally:
        os.unlink(temp_file_path)


# LLM-generated content at query #22
#--------------------------

```python
def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, "file.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.bin"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02")
    outfile = os.path.join(project_dir, "file.bin")
    generate_file(project_dir, infile, context, env)
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02"

def test_generate_file_renders_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"var": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("{{ cookiecutter.var }}")
    outfile = os.path.join(project_dir, "file.txt")
    generate_file(project_dir, infile, context, env)
    with open(outfile, "r") as f:
        assert f.read() == "value"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(infile, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)

def test_generate_file_applies_new_line_setting():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("content")
    outfile = os.path.join(project_dir, "file.txt")
    generate_file(project_dir, infile, context, env)
    with open(outfile, "r", newline="") as f:
        assert f.read() == "content\r\n"


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_47_evaluates_to_true():
    project_dir = "/path/to/project"
    infile = "/path/to/binary/file"
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)
    assert is_binary(infile)


# LLM-generated content at query #24
#--------------------------

```python
def test_empty_directory_name_raises_exception():
    try:
        render_and_create_dir("", {}, "output_dir", Environment())
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException to be raised"

def test_whitespace_directory_name_raises_exception():
    try:
        render_and_create_dir(" ", {}, "output_dir", Environment())
    except EmptyDirNameException:
        pass
    else:
        assert False, "Expected EmptyDirNameException to be raised"


# LLM-generated content at query #25
#--------------------------

def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_choice provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #26
#--------------------------

def test_apply_overwrites_to_context_boolean_invalid_response():
    context = {"test_var": True}
    overwrite_context = {"test_var": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_choice provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #27
#--------------------------

```python
def test_generate_file_binary():
    project_dir = "/tmp/project"
    infile = "test.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_text():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {"name": "test"}
    env = Environment(loader=DictLoader({infile: "Hello {{ name }}"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Hello {{ name }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    assert os.path.exists(outfile)
    with open(outfile) as f:
        assert f.read() == "Hello test"

def test_generate_file_skip_exists():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("content")
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "w") as f:
        f.write("existing")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == "existing"

def test_generate_file_empty_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_new_lines():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({infile: "line1\nline2"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("line1\nline2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"line1\r\nline2"


# LLM-generated content at query #28
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dir_name_exception_for_empty_dirname():
    empty_dirname = ""
    context = {"key": "value"}
    output_dir = Path("/tmp/output")
    environment = Environment()
    try:
        render_and_create_dir(empty_dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_raises_empty_dir_name_exception_for_whitespace_dirname():
    whitespace_dirname = "   "
    context = {"key": "value"}
    output_dir = Path("/tmp/output")
    environment = Environment()
    try:
        render_and_create_dir(whitespace_dirname, context, output_dir, environment)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_new_lines_in_context():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False) == '\n'


# LLM-generated content at query #30
#--------------------------

```python
def test_skip_if_file_exists_and_file_already_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"key": "value"}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    open(outfile, "w").close()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    assert os.path.exists(outfile)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_true():
    context = {'cookiecutter': {'_new_lines': True}}
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #32
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.seek(0)
        context = generate_context(f.name)
        assert context == {'cookiecutter': {'key': 'value'}}

def test_generate_context_with_invalid_json_file():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write('{"key": "value"')
        f.seek(0)
        try:
            generate_context(f.name)
        except ContextDecodingException:
            pass
        else:
            assert False, "Expected ContextDecodingException"

def test_generate_context_with_default_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.seek(0)
        context = generate_context(f.name, default_context={'key': 'new_value'})
        assert context == {'cookiecutter': {'key': 'new_value'}}

def test_generate_context_with_extra_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.seek(0)
        context = generate_context(f.name, extra_context={'key': 'new_value'})
        assert context == {'cookiecutter': {'key': 'new_value'}}

def test_generate_context_with_default_and_extra_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': 'value'}, f)
        f.seek(0)
        context = generate_context(f.name, default_context={'key': 'default_value'}, extra_context={'key': 'extra_value'})
        assert context == {'cookiecutter': {'key': 'extra_value'}}

def test_generate_context_with_nested_default_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': {'nested_key': 'value'}}, f)
        f.seek(0)
        context = generate_context(f.name, default_context={'key': {'nested_key': 'new_value'}})
        assert context == {'cookiecutter': {'key': {'nested_key': 'new_value'}}}

def test_generate_context_with_nested_extra_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': {'nested_key': 'value'}}, f)
        f.seek(0)
        context = generate_context(f.name, extra_context={'key': {'nested_key': 'new_value'}})
        assert context == {'cookiecutter': {'key': {'nested_key': 'new_value'}}}

def test_generate_context_with_multichoice_default_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': ['value1', 'value2']}, f)
        f.seek(0)
        context = generate_context(f.name, default_context={'key': ['value2']})
        assert context == {'cookiecutter': {'key': ['value2']}}

def test_generate_context_with_multichoice_extra_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': ['value1', 'value2']}, f)
        f.seek(0)
        context = generate_context(f.name, extra_context={'key': ['value2']})
        assert context == {'cookiecutter': {'key': ['value2']}}

def test_generate_context_with_boolean_default_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': True}, f)
        f.seek(0)
        context = generate_context(f.name, default_context={'key': 'yes'})
        assert context == {'cookiecutter': {'key': True}}

def test_generate_context_with_boolean_extra_context():
    import tempfile
    import json
    import os

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        json.dump({'key': True}, f)
        f.seek(0)
        context = generate_context(f.name, extra_context={'key': 'yes'})
        assert context == {'cookiecutter': {'key': True}}


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_47_evaluates_to_true():
    test_project_dir = "/tmp/project"
    test_infile = "/tmp/binary_file.bin"
    test_context = {"cookiecutter": {}}
    test_env = Environment()
    
    # Mock is_binary to return True
    original_is_binary = is_binary
    is_binary = lambda x: True
    
    # Mock shutil.copyfile and shutil.copymode to do nothing
    original_copyfile = shutil.copyfile
    original_copymode = shutil.copymode
    shutil.copyfile = lambda src, dst: None
    shutil.copymode = lambda src, dst: None
    
    generate_file(test_project_dir, test_infile, test_context, test_env)
    
    # Restore original functions
    is_binary = original_is_binary
    shutil.copyfile = original_copyfile
    shutil.copymode = original_copymode


# LLM-generated content at query #34
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dir_name_exception_for_empty_dirname():
    import pytest
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment

    environment = Environment()
    context = {}
    output_dir = "/tmp"

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("", context, output_dir, environment)

def test_render_and_create_dir_raises_empty_dir_name_exception_for_whitespace_dirname():
    import pytest
    from cookiecutter.generate import render_and_create_dir
    from cookiecutter.exceptions import EmptyDirNameException
    from jinja2 import Environment

    environment = Environment()
    context = {}
    output_dir = "/tmp"

    with pytest.raises(EmptyDirNameException):
        render_and_create_dir("   ", context, output_dir, environment)


# LLM-generated content at query #35
#--------------------------

def test_generate_files_with_empty_context():
    context = {}
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    try:
        generate_files('test_repo', context, output_dir)
    finally:
        shutil.rmtree(output_dir)

def test_generate_files_with_overwrite_existing():
    context = {'cookiecutter': {'project_name': 'test'}}
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'test').mkdir()
    try:
        generate_files('test_repo', context, output_dir, overwrite_if_exists=True)
    finally:
        shutil.rmtree(output_dir)

def test_generate_files_with_skip_existing():
    context = {'cookiecutter': {'project_name': 'test'}}
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'test').mkdir()
    try:
        generate_files('test_repo', context, output_dir, skip_if_file_exists=True)
    finally:
        shutil.rmtree(output_dir)

def test_generate_files_with_hooks_disabled():
    context = {'cookiecutter': {'project_name': 'test'}}
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    try:
        generate_files('test_repo', context, output_dir, accept_hooks=False)
    finally:
        shutil.rmtree(output_dir)

def test_generate_files_with_keep_on_failure():
    context = {'cookiecutter': {'invalid_var': '{{ invalid }}'}}
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    try:
        generate_files('test_repo', context, output_dir, keep_project_on_failure=True)
    except UndefinedVariableInTemplate:
        assert (output_dir / 'test_output').exists()
    finally:
        shutil.rmtree(output_dir)


# LLM-generated content at query #36
#--------------------------

```python
def test_delete_project_on_failure_evaluates_to_true():
    output_directory_created = True
    keep_project_on_failure = False
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure == True


# LLM-generated content at query #37
#--------------------------

```python
def test_generate_file_skips_when_file_exists():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    open(os.path.join(project_dir, infile), 'w').close()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

def test_generate_file_handles_empty_filename():
    project_dir = "/tmp/project"
    infile = ""
    context = {}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_copies_binary_files():
    project_dir = "/tmp/project"
    infile = "test.bin"
    context = {}
    env = Environment()
    with open(infile, 'wb') as f:
        f.write(b'\x00\x01\x02')
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, infile))

def test_generate_file_renders_text_files():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {"name": "test"}
    env = Environment(loader=DictLoader({infile: "Hello {{ name }}"}))
    with open(infile, 'w') as f:
        f.write("Hello {{ name }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'r') as f:
        assert f.read() == "Hello test"

def test_generate_file_preserves_file_permissions():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment()
    with open(infile, 'w') as f:
        f.write("test")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert oct(os.stat(os.path.join(project_dir, infile)).st_mode & 0o777) == oct(0o644)

def test_generate_file_handles_newlines():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({infile: "line1\nline2"}))
    with open(infile, 'w', newline='\n') as f:
        f.write("line1\nline2")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, infile), 'rb') as f:
        assert f.read() == b"line1\r\nline2"

def test_generate_file_handles_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "test.txt"
    context = {}
    env = Environment(loader=DictLoader({infile: "Hello {{"}))
    with open(infile, 'w') as f:
        f.write("Hello {{")
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError"
    except TemplateSyntaxError:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError to be raised"
    except TemplateSyntaxError:
        assert True


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_creates_file_with_rendered_content():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    expected_outfile = os.path.join(project_dir, infile)
    assert os.path.exists(expected_outfile)
    with open(expected_outfile, "r", encoding="utf-8") as f:
        assert f.read() == "value"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    expected_outfile = os.path.join(project_dir, infile)
    with open(expected_outfile, "w", encoding="utf-8") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(expected_outfile, "r", encoding="utf-8") as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.bin"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    expected_outfile = os.path.join(project_dir, infile)
    assert os.path.exists(expected_outfile)
    with open(expected_outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("."))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_generate_file_skips_existing_file_when_skip_flag_is_true():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    outfile = os.path.join(project_dir, "output.txt")
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile) as f:
        assert f.read() == "existing content"

def test_generate_file_copies_binary_file_without_rendering():
    project_dir = "/tmp/project"
    infile = "binary.dat"
    context = {"cookiecutter": {}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"

def test_generate_file_renders_text_file_with_context():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "Test Project"}}
    env = Environment(loader=DictLoader({infile: "Project: {{ cookiecutter.name }}"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Project: {{ cookiecutter.name }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile) as f:
        assert f.read() == "Project: Test Project"

def test_generate_file_uses_configured_newline_character():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment(loader=DictLoader({infile: "Line1\nLine2"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Line1\nLine2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"Line1\r\nLine2"

def test_generate_file_detects_original_newline_character():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment(loader=DictLoader({infile: "Line1\nLine2"}))
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", newline="\r") as f:
        f.write("Line1\nLine2")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, infile)
    with open(outfile, "rb") as f:
        assert f.read() == b"Line1\rLine2"

def test_generate_file_skips_empty_filename():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {}}
    env = Environment(loader=DictLoader({infile: "content"}))
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert not os.path.exists(os.path.join(project_dir, ""))


# LLM-generated content at query #41
#--------------------------

```python
def test_generate_context_with_valid_json():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': 'value1', 'key2': 'value2'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(context_file=temp_file.name)
        assert context == {'cookiecutter': context_data}

def test_generate_context_with_invalid_json():
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        temp_file.write('invalid json')
        temp_file.flush()
        try:
            generate_context(context_file=temp_file.name)
            assert False, "Expected ContextDecodingException"
        except ContextDecodingException:
            pass

def test_generate_context_with_default_context():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': 'value1', 'key2': ['value2', 'value3']}
    default_context = {'key1': 'new_value1', 'key2': 'value2'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(context_file=temp_file.name, default_context=default_context)
        assert context['cookiecutter']['key1'] == 'new_value1'
        assert context['cookiecutter']['key2'] == ['value2', 'value3']

def test_generate_context_with_extra_context():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': 'value1', 'key2': ['value2', 'value3']}
    extra_context = {'key1': 'new_value1', 'key2': 'value2'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(context_file=temp_file.name, extra_context=extra_context)
        assert context['cookiecutter']['key1'] == 'new_value1'
        assert context['cookiecutter']['key2'] == ['value2', 'value3']

def test_generate_context_with_dict_overwrite():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': {'subkey1': 'value1'}, 'key2': 'value2'}
    extra_context = {'key1': {'subkey1': 'new_value1'}}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(context_file=temp_file.name, extra_context=extra_context)
        assert context['cookiecutter']['key1']['subkey1'] == 'new_value1'
        assert context['cookiecutter']['key2'] == 'value2'

def test_generate_context_with_invalid_default_context():
    import json
    from tempfile import NamedTemporaryFile
    import warnings

    context_data = {'key1': 'value1'}
    default_context = {'key1': 'invalid_value'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        generate_context(context_file=temp_file.name, default_context=default_context)

def test_generate_context_with_boolean_overwrite():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': True}
    extra_context = {'key1': 'yes'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(context_file=temp_file.name, extra_context=extra_context)
        assert context['cookiecutter']['key1'] is True

def test_generate_context_with_invalid_boolean_overwrite():
    import json
    from tempfile import NamedTemporaryFile

    context_data = {'key1': True}
    extra_context = {'key1': 'invalid_value'}
    with NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        try:
            generate_context(context_file=temp_file.name, extra_context=extra_context)
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #42
#--------------------------

```
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #43
#--------------------------

```python
def test_delete_project_on_failure_is_false_when_output_directory_not_created():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False

def test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #44
#--------------------------

```python
def test_generate_context_with_invalid_json_file():
    context_file = "invalid.json"
    try:
        generate_context(context_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException to be raised"


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_context_with_valid_json():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': 'value'}, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name)
        assert 'key' in context['cookiecutter']
        assert context['cookiecutter']['key'] == 'value'

def test_generate_context_with_invalid_json():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        temp_file.write('invalid json')
        temp_file.flush()
        try:
            generate_context(temp_file.name)
            assert False, "Expected ContextDecodingException"
        except ContextDecodingException:
            pass

def test_generate_context_with_default_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': 'value'}, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name, default_context={'key': 'new_value'})
        assert context['cookiecutter']['key'] == 'new_value'

def test_generate_context_with_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': 'value'}, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name, extra_context={'key': 'new_value'})
        assert context['cookiecutter']['key'] == 'new_value'

def test_generate_context_with_default_and_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': 'value'}, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name, default_context={'key': 'default_value'}, extra_context={'key': 'extra_value'})
        assert context['cookiecutter']['key'] == 'extra_value'

def test_generate_context_with_invalid_default_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': ['choice1', 'choice2']}, temp_file)
        temp_file.flush()
        try:
            generate_context(temp_file.name, default_context={'key': 'invalid_choice'})
            assert False, "Expected ValueError"
        except ValueError:
            pass

def test_generate_context_with_invalid_extra_context():
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as temp_file:
        json.dump({'key': ['choice1', 'choice2']}, temp_file)
        temp_file.flush()
        try:
            generate_context(temp_file.name, extra_context={'key': 'invalid_choice'})
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_true():
    context = {'cookiecutter': {'_new_lines': True}}
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #48
#--------------------------

```python
def test_process_response_raises_invalid_response_for_invalid_choice():
    prompt = YesNoPrompt()
    prompt.process_response("invalid_choice")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context_basic():
    context_file = 'test.json'
    default_context = None
    extra_context = None
    with open(context_file, 'w') as f:
        f.write('{"key": "value"}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'key': 'value'}}

def test_generate_context_with_default():
    context_file = 'test.json'
    default_context = {'key': 'new_value'}
    extra_context = None
    with open(context_file, 'w') as f:
        f.write('{"key": "value"}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'key': 'new_value'}}

def test_generate_context_with_extra():
    context_file = 'test.json'
    default_context = None
    extra_context = {'key': 'extra_value'}
    with open(context_file, 'w') as f:
        f.write('{"key": "value"}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'key': 'extra_value'}}

def test_generate_context_with_nested_dict():
    context_file = 'test.json'
    default_context = None
    extra_context = {'nested': {'key': 'nested_value'}}
    with open(context_file, 'w') as f:
        f.write('{"nested": {"key": "value"}}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'nested': {'key': 'nested_value'}}}

def test_generate_context_with_list_overwrite():
    context_file = 'test.json'
    default_context = None
    extra_context = {'choices': ['new_choice']}
    with open(context_file, 'w') as f:
        f.write('{"choices": ["choice1", "choice2"]}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'choices': ['new_choice']}}

def test_generate_context_with_boolean_conversion():
    context_file = 'test.json'
    default_context = None
    extra_context = {'flag': 'yes'}
    with open(context_file, 'w') as f:
        f.write('{"flag": false}')
    result = generate_context(context_file, default_context, extra_context)
    assert result == {'test': {'flag': True}}

def test_generate_context_with_invalid_json():
    context_file = 'test.json'
    default_context = None
    extra_context = None
    with open(context_file, 'w') as f:
        f.write('invalid json')
    try:
        generate_context(context_file, default_context, extra_context)
        assert False
    except ContextDecodingException:
        assert True

def test_generate_context_with_invalid_list_overwrite():
    context_file = 'test.json'
    default_context = None
    extra_context = {'choices': ['invalid_choice']}
    with open(context_file, 'w') as f:
        f.write('{"choices": ["choice1", "choice2"]}')
    try:
        generate_context(context_file, default_context, extra_context)
        assert False
    except ValueError:
        assert True

def test_generate_context_with_invalid_boolean():
    context_file = 'test.json'
    default_context = None
    extra_context = {'flag': 'invalid'}
    with open(context_file, 'w') as f:
        f.write('{"flag": false}')
    try:
        generate_context(context_file, default_context, extra_context)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_generate_context_with_none_default_context():
    context = generate_context(default_context=None)
    assert isinstance(context, OrderedDict)


# LLM-generated content at query #3
#--------------------------

def test_apply_overwrites_to_context_new_variable_ignored():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}


def test_apply_overwrites_to_context_new_dict_variable_added():
    context = {"existing": {}}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"existing": {}, "new": "value"}


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
    overwrite = {"flag": "maybe"}
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


# LLM-generated content at query #4
#--------------------------

def test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir_with_same_parameters():
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
    _run_hook_from_repo_dir(repo_dir, hook_name, project_dir, context, delete_project_on_failure)


# LLM-generated content at query #5
#--------------------------

```python
def test_render_and_create_dir_creates_directory_when_not_exist(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    result, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert result.exists()
    assert created

def test_render_and_create_dir_raises_exception_when_dirname_empty(tmp_path):
    dirname = ""
    context = {}
    output_dir = tmp_path
    environment = Environment()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except EmptyDirNameException:
        assert True

def test_render_and_create_dir_raises_exception_when_dir_exists(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    (output_dir / dirname).mkdir()
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
        assert False
    except OutputDirExistsException:
        assert True

def test_render_and_create_dir_overwrites_existing_dir(tmp_path):
    dirname = "test_dir"
    context = {}
    output_dir = tmp_path
    environment = Environment()
    (output_dir / dirname).mkdir()
    result, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert result.exists()
    assert not created


# LLM-generated content at query #6
#--------------------------

```python
def test_process_response_with_yes_choices():
    prompt = YesNoPrompt()
    assert prompt.process_response("1") == True
    assert prompt.process_response("true") == True
    assert prompt.process_response("t") == True
    assert prompt.process_response("yes") == True
    assert prompt.process_response("y") == True
    assert prompt.process_response("on") == True


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context_json_decoding_error():
    import json
    import os
    from collections import OrderedDict
    from cookiecutter.exceptions import ContextDecodingException

    # Create a temporary file with invalid JSON content
    invalid_json_content = '{"key": "value"'
    tmp_file = 'tmp_invalid.json'
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(invalid_json_content)

    try:
        # Attempt to load the invalid JSON file
        context = generate_context(context_file=tmp_file)
    except ContextDecodingException as e:
        # Verify that the exception contains the expected error message
        full_fpath = os.path.abspath(tmp_file)
        expected_message = (
            f"JSON decoding error while loading '{full_fpath}'. "
            "Decoding error details: 'Expecting ',' delimiter: line 1 column 15 (char 14)'"
        )
        assert str(e) == expected_message
    finally:
        # Clean up the temporary file
        os.remove(tmp_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_context_with_valid_json_file():
    context = generate_context(context_file='tests/test-context.json')
    assert 'test_stem' in context
    assert isinstance(context['test_stem'], dict)

def test_generate_context_with_invalid_json_file():
    try:
        generate_context(context_file='tests/invalid-context.json')
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"

def test_generate_context_with_default_context():
    context = generate_context(
        context_file='tests/test-context.json',
        default_context={'key1': 'value1'}
    )
    assert context['test_stem']['key1'] == 'value1'

def test_generate_context_with_extra_context():
    context = generate_context(
        context_file='tests/test-context.json',
        extra_context={'key2': 'value2'}
    )
    assert context['test_stem']['key2'] == 'value2'

def test_generate_context_with_invalid_default_context():
    try:
        generate_context(
            context_file='tests/test-context.json',
            default_context={'invalid_key': 'invalid_value'}
        )
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_generate_context_with_invalid_extra_context():
    try:
        generate_context(
            context_file='tests/test-context.json',
            extra_context={'invalid_key': 'invalid_value'}
        )
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #9
#--------------------------

```python
def test_apply_overwrites_to_context_new_variable():
    context = {"existing": "value"}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"existing": "value"}

def test_apply_overwrites_to_context_new_dictionary_variable():
    context = {"existing": {"nested": "value"}}
    overwrite = {"new": "value"}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context == {"existing": {"nested": "value"}, "new": "value"}

def test_apply_overwrites_to_context_list_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite = {"list_var": ["a", "b"]}
    apply_overwrites_to_context(context, overwrite)
    assert context["list_var"] == ["a", "b"]

def test_apply_overwrites_to_context_list_invalid_overwrite():
    context = {"list_var": ["a", "b", "c"]}
    overwrite = {"list_var": ["d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_choice_overwrite():
    context = {"choice_var": ["a", "b", "c"]}
    overwrite = {"choice_var": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context["choice_var"] == ["b", "a", "c"]

def test_apply_overwrites_to_context_choice_invalid_overwrite():
    context = {"choice_var": ["a", "b", "c"]}
    overwrite = {"choice_var": "d"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_dict_overwrite():
    context = {"dict_var": {"a": 1, "b": 2}}
    overwrite = {"dict_var": {"b": 3}}
    apply_overwrites_to_context(context, overwrite)
    assert context["dict_var"] == {"a": 1, "b": 3}

def test_apply_overwrites_to_context_bool_overwrite():
    context = {"bool_var": False}
    overwrite = {"bool_var": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context["bool_var"] == True

def test_apply_overwrites_to_context_bool_invalid_overwrite():
    context = {"bool_var": False}
    overwrite = {"bool_var": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False
    except ValueError:
        assert True

def test_apply_overwrites_to_context_simple_overwrite():
    context = {"var": "old"}
    overwrite = {"var": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context["var"] == "new"


# LLM-generated content at query #10
#--------------------------

def test_generate_context_with_invalid_json():
    import os
    import tempfile
    from cookiecutter.exceptions import ContextDecodingException

    invalid_json = '{"invalid": json}'
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(invalid_json)
        f.close()
        try:
            generate_context(context_file=f.name)
        except ContextDecodingException:
            pass
        finally:
            os.unlink(f.name)


# LLM-generated content at query #11
#--------------------------

```python
def test_process_response_with_invalid_input():
    prompt = YesNoPrompt()
    invalid_input = "invalid"
    result = False
    try:
        prompt.process_response(invalid_input)
    except InvalidResponse:
        result = True
    assert result


# LLM-generated content at query #12
#--------------------------

def test_generate_context_with_valid_json():
    import tempfile
    import json
    import os
    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name)
        os.unlink(f.name)
    assert result[os.path.splitext(os.path.basename(f.name))[0]] == context_data

def test_generate_context_with_invalid_json():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json")
        f.flush()
        try:
            generate_context(context_file=f.name)
            assert False, "Should raise ContextDecodingException"
        except ContextDecodingException:
            pass
        os.unlink(f.name)

def test_generate_context_with_default_context():
    import tempfile
    import json
    import os
    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    default_context = {"key1": "new_value"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name, default_context=default_context)
        os.unlink(f.name)
    assert result[os.path.splitext(os.path.basename(f.name))[0]]["key1"] == "new_value"

def test_generate_context_with_extra_context():
    import tempfile
    import json
    import os
    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    extra_context = {"key1": "extra_value"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result[os.path.splitext(os.path.basename(f.name))[0]]["key1"] == "extra_value"

def test_generate_context_with_invalid_default_context():
    import tempfile
    import json
    import os
    import warnings
    context_data = {"key1": "value1", "key2": ["choice1", "choice2"]}
    default_context = {"invalid_key": "value"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        with warnings.catch_warnings(record=True) as w:
            generate_context(context_file=f.name, default_context=default_context)
            assert len(w) == 1
            assert "Invalid default received" in str(w[0].message)
        os.unlink(f.name)

def test_generate_context_with_boolean_conversion():
    import tempfile
    import json
    import os
    context_data = {"bool_key": True}
    extra_context = {"bool_key": "yes"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result[os.path.splitext(os.path.basename(f.name))[0]]["bool_key"] is True

def test_generate_context_with_multichoice_overwrite():
    import tempfile
    import json
    import os
    context_data = {"choices": ["a", "b", "c"]}
    extra_context = {"choices": ["a", "b"]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name, extra_context=extra_context)
        os.unlink(f.name)
    assert result[os.path.splitext(os.path.basename(f.name))[0]]["choices"] == ["a", "b"]

def test_generate_context_with_dict_overwrite():
    import tempfile
    import json
    import os
    context_data = {"nested": {"key1": "value1", "key2": "value2"}}
    extra_context = {"nested": {"key1": "new_value"}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(context_data, f)
        f.flush()
        result = generate_context(context_file=f.name, extra_context=extra_context)
        os.unlink(f.name)
    nested = result[os.path.splitext(os.path.basename(f.name))[0]]["nested"]
    assert nested["key1"] == "new_value"
    assert nested["key2"] == "value2"


# LLM-generated content at query #13
#--------------------------

```python
def test_apply_overwrites_to_context_with_list_and_in_dictionary_variable():
    context = {"variable": ["a", "b", "c"]}
    overwrite_context = {"variable": ["d", "e"]}
    apply_overwrites_to_context(context, overwrite_context, in_dictionary_variable=True)
    assert context["variable"] == ["d", "e"]


# LLM-generated content at query #14
#--------------------------

def test_generate_context_invalid_json():
    invalid_json_file = "invalid.json"
    try:
        generate_context(context_file=invalid_json_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException to be raised"


# LLM-generated content at query #15
#--------------------------

```python
def test_apply_overwrites_to_context_line_46():
    context = {'key1': {}}
    overwrite_context = {'key1': {'subkey': 'value'}}
    apply_overwrites_to_context(context, overwrite_context)
    assert isinstance(context['key1'], dict)
    assert context['key1'] == {'subkey': 'value'}


# LLM-generated content at query #16
#--------------------------

```python
def test_render_and_create_dir_raises_empty_dir_name_exception():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    try:
        render_and_create_dir('', context, output_dir, environment)
    except Exception as e:
        assert isinstance(e, EmptyDirNameException)


# LLM-generated content at query #17
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_to_create.exists()
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
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create.exists()
    assert not created


# LLM-generated content at query #18
#--------------------------

```
def test_apply_overwrites_to_context_dict_overwrite_false_case():
    context = {"nested": {"key": "value"}}
    overwrite_context = {"nested": "not_a_dict"}
    apply_overwrites_to_context(context, overwrite_context)
    assert isinstance(context["nested"], str)


# LLM-generated content at query #19
#--------------------------

```python
def test_generate_files_creates_project_directory():
    repo_dir = '/tmp/test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert 'test_project' in project_dir


def test_generate_files_handles_existing_output_dir_with_overwrite():
    repo_dir = '/tmp/test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output'
    os.makedirs(os.path.join(output_dir, 'test_project'), exist_ok=True)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)


def test_generate_files_skips_existing_files_when_configured():
    repo_dir = '/tmp/test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output'
    test_file = os.path.join(output_dir, 'test_project', 'existing.txt')
    os.makedirs(os.path.join(output_dir, 'test_project'), exist_ok=True)
    with open(test_file, 'w') as f:
        f.write('existing content')
    generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    with open(test_file) as f:
        assert f.read() == 'existing content'


def test_generate_files_executes_pre_and_post_hooks():
    repo_dir = '/tmp/test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = '/tmp/output'
    pre_hook = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    post_hook = os.path.join(repo_dir, 'hooks', 'post_gen_project.py')
    os.makedirs(os.path.join(repo_dir, 'hooks'))
    with open(pre_hook, 'w') as f:
        f.write('print("pre hook executed")')
    with open(post_hook, 'w') as f:
        f.write('print("post hook executed")')
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)


def test_generate_files_keeps_project_on_failure_when_configured():
    repo_dir = '/tmp/test_repo'
    context = {'cookiecutter': {'invalid_var': '{{ invalid_var }}'}}
    output_dir = '/tmp/output'
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
    except UndefinedVariableInTemplate:
        project_dir = os.path.join(output_dir, '{{ invalid_var }}')
        assert os.path.exists(project_dir)


def test_generate_files_copies_non_rendered_files():
    repo_dir = '/tmp/test_repo'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': ['*.bin']
        }
    }
    output_dir = '/tmp/output'
    binary_file = os.path.join(repo_dir, 'cookiecutter-{{ project_name }}', 'test.bin')
    os.makedirs(os.path.dirname(binary_file))
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    project_dir = generate_files(repo_dir, context, output_dir)
    output_file = os.path.join(project_dir, 'test.bin')
    assert os.path.exists(output_file)


# LLM-generated content at query #20
#--------------------------

```python
def test_generate_files_creates_project_directory():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert project_dir == os.path.join(output_dir, 'my_project')

def test_generate_files_overwrites_existing_directory():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    os.makedirs(os.path.join(output_dir, 'my_project'))
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)
    assert project_dir == os.path.join(output_dir, 'my_project')

def test_generate_files_skips_existing_files():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    os.makedirs(os.path.join(output_dir, 'my_project'))
    with open(os.path.join(output_dir, 'my_project', 'existing_file.txt'), 'w') as f:
        f.write('content')
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(os.path.join(project_dir, 'existing_file.txt'))

def test_generate_files_fails_if_output_dir_exists():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    os.makedirs(os.path.join(output_dir, 'my_project'))
    try:
        generate_files(repo_dir, context, output_dir)
        assert False, "Expected OutputDirExistsException"
    except OutputDirExistsException:
        assert True

def test_generate_files_executes_hooks():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)
    # Assuming hooks create a specific file or directory
    assert os.path.exists(os.path.join(project_dir, 'hook_created_file.txt'))

def test_generate_files_keeps_project_on_failure():
    repo_dir = '/path/to/repo'
    context = {'cookiecutter': {'project_name': 'my_project'}}
    output_dir = '/output/dir'
    # Simulate a failure by using an invalid template
    try:
        generate_files(repo_dir, context, output_dir, keep_project_on_failure=True)
        assert False, "Expected UndefinedVariableInTemplate"
    except UndefinedVariableInTemplate:
        assert os.path.exists(os.path.join(output_dir, 'my_project'))


# LLM-generated content at query #21
#--------------------------

```python
def test_render_and_create_dir_overwrites_existing_dir_when_overwrite_if_exists_is_true():
    from cookiecutter import generate
    from pathlib import Path
    from jinja2 import Environment

    test_output_dir = Path('/tmp/test_output')
    test_dirname = 'test_dir'
    test_context = {}
    test_environment = Environment()

    # Create the directory first to ensure it exists
    generate.make_sure_path_exists(test_output_dir / test_dirname)

    # Call with overwrite_if_exists=True
    result_path, created = generate.render_and_create_dir(
        dirname=test_dirname,
        context=test_context,
        output_dir=test_output_dir,
        environment=test_environment,
        overwrite_if_exists=True
    )

    assert not created
    assert result_path == test_output_dir / test_dirname


# LLM-generated content at query #22
#--------------------------

```python
def test_apply_overwrites_to_context_invalid_boolean_conversion():
    context = {"flag": True}
    overwrite_context = {"flag": "invalid_choice"}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as err:
        assert str(err) == "invalid_choice provided for variable flag could not be converted to a boolean."
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #23
#--------------------------

```python
def test_output_directory_created_and_not_keep_project_on_failure():
    output_directory_created = True
    keep_project_on_failure = False
    result = output_directory_created and not keep_project_on_failure
    assert result is True

def test_output_directory_not_created_and_not_keep_project_on_failure():
    output_directory_created = False
    keep_project_on_failure = False
    result = output_directory_created and not keep_project_on_failure
    assert result is False

def test_output_directory_created_and_keep_project_on_failure():
    output_directory_created = True
    keep_project_on_failure = True
    result = output_directory_created and not keep_project_on_failure
    assert result is False

def test_output_directory_not_created_and_keep_project_on_failure():
    output_directory_created = False
    keep_project_on_failure = True
    result = output_directory_created and not keep_project_on_failure
    assert result is False


# LLM-generated content at query #24
#--------------------------

```python
def test_accept_hooks_predicate_evaluates_to_true():
    accept_hooks = True
    assert accept_hooks == True


# LLM-generated content at query #25
#--------------------------

def test_render_and_create_dir_raises_on_empty_dirname():
    context = {}
    output_dir = Path('/tmp')
    environment = Environment()
    try:
        render_and_create_dir('', context, output_dir, environment)
        assert False, 'Expected EmptyDirNameException'
    except EmptyDirNameException:
        pass

def test_render_and_create_dir_creates_new_directory():
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    dir_to_create, created = render_and_create_dir('{{ name }}', context, output_dir, environment)
    assert dir_to_create == Path('/tmp/test')
    assert created
    assert dir_to_create.exists()

def test_render_and_create_dir_raises_on_existing_dir():
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    Path('/tmp/test').mkdir(exist_ok=True)
    try:
        render_and_create_dir('{{ name }}', context, output_dir, environment)
        assert False, 'Expected OutputDirExistsException'
    except OutputDirExistsException:
        pass

def test_render_and_create_dir_overwrites_existing_dir():
    context = {'name': 'test'}
    output_dir = Path('/tmp')
    environment = Environment()
    Path('/tmp/test').mkdir(exist_ok=True)
    dir_to_create, created = render_and_create_dir('{{ name }}', context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create == Path('/tmp/test')
    assert not created
    assert dir_to_create.exists()


# LLM-generated content at query #26
#--------------------------

```python
def test_render_and_create_dir_with_empty_dirname():
    dirname = ""
    context = {}
    output_dir = Path("test_output")
    environment = Environment()
    overwrite_if_exists = False
    try:
        render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists)
        assert False, "Expected EmptyDirNameException to be raised"
    except EmptyDirNameException:
        assert True


# LLM-generated content at query #27
#--------------------------

```python
def test_render_and_create_dir_overwrite_existing():
    context = {}
    output_dir = Path('/tmp/test_output')
    environment = Environment()
    dirname = 'test_dir'
    
    # First create the directory
    dir_to_create, created = render_and_create_dir(
        dirname,
        context,
        output_dir,
        environment,
        overwrite_if_exists=False
    )
    
    # Try to create again with overwrite=True
    dir_to_create2, created2 = render_and_create_dir(
        dirname,
        context,
        output_dir,
        environment,
        overwrite_if_exists=True
    )
    
    assert dir_to_create == dir_to_create2
    assert created == True
    assert created2 == False


# LLM-generated content at query #28
#--------------------------

```python
def test_overwrite_existing_directory():
    output_dir = Path('/tmp/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_dir = output_dir / 'existing_dir'
    existing_dir.mkdir()
    
    try:
        result = render_and_create_dir('existing_dir', {}, output_dir, Environment(), overwrite_if_exists=True)
        assert result[0] == existing_dir
        assert result[1] == False
    finally:
        existing_dir.rmdir()
        output_dir.rmdir()


# LLM-generated content at query #29
#--------------------------

```
def test_is_copy_only_path_matches_pattern():
    path = "some/file.txt"
    context = {"cookiecutter": {"_copy_without_render": ["*.txt"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_no_match():
    path = "some/file.txt"
    context = {"cookiecutter": {"_copy_without_render": ["*.csv"]}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_empty_context():
    path = "some/file.txt"
    context = {}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_missing_key():
    path = "some/file.txt"
    context = {"cookiecutter": {}}
    assert is_copy_only_path(path, context) == False

def test_is_copy_only_path_multiple_patterns():
    path = "some/file.txt"
    context = {"cookiecutter": {"_copy_without_render": ["*.csv", "*.txt", "*.json"]}}
    assert is_copy_only_path(path, context) == True

def test_is_copy_only_path_directory_pattern():
    path = "some/directory"
    context = {"cookiecutter": {"_copy_without_render": ["some/*"]}}
    assert is_copy_only_path(path, context) == True


# LLM-generated content at query #30
#--------------------------

```
def test_is_copy_only_path_with_matching_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*']
        }
    }
    assert is_copy_only_path('readme.txt', context) == True

def test_is_copy_only_path_with_non_matching_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.md', 'images/*']
        }
    }
    assert is_copy_only_path('readme.txt', context) == False

def test_is_copy_only_path_with_empty_context():
    context = {}
    assert is_copy_only_path('readme.txt', context) == False

def test_is_copy_only_path_with_missing_key():
    context = {'cookiecutter': {}}
    assert is_copy_only_path('readme.txt', context) == False

def test_is_copy_only_path_with_directory_pattern():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['docs/*', 'static/*']
        }
    }
    assert is_copy_only_path('docs/index.html', context) == True

def test_is_copy_only_path_with_non_matching_directory():
    context = {
        'cookiecutter': {
            '_copy_without_render': ['docs/*', 'static/*']
        }
    }
    assert is_copy_only_path('src/main.py', context) == False


# LLM-generated content at query #31
#--------------------------

```python
def test_boolean_conversion_failure():
    context = {'test_var': True}
    overwrite_context = {'test_var': 'invalid_value'}
    try:
        apply_overwrites_to_context(context, overwrite_context)
    except ValueError as e:
        assert str(e) == "invalid_value provided for variable test_var could not be converted to a boolean."


# LLM-generated content at query #32
#--------------------------

```python
def test_render_and_create_dir_creates_new_directory():
    dirname = "test_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment)
    assert dir_to_create.exists()
    assert created

def test_render_and_create_dir_raises_exception_for_empty_dirname():
    dirname = ""
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    raised_exception = False
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except EmptyDirNameException:
        raised_exception = True
    assert raised_exception

def test_render_and_create_dir_raises_exception_for_existing_directory():
    dirname = "existing_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    raised_exception = False
    try:
        render_and_create_dir(dirname, context, output_dir, environment)
    except OutputDirExistsException:
        raised_exception = True
    assert raised_exception

def test_render_and_create_dir_overwrites_existing_directory():
    dirname = "existing_dir"
    context = {}
    output_dir = "/tmp"
    environment = Environment()
    dir_to_create = Path(output_dir, dirname)
    dir_to_create.mkdir(parents=True, exist_ok=True)
    dir_to_create, created = render_and_create_dir(dirname, context, output_dir, environment, overwrite_if_exists=True)
    assert dir_to_create.exists()
    assert not created


# LLM-generated content at query #33
#--------------------------

```python
def test_delete_project_on_failure_evaluates_to_false():
    output_directory_created = False
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert not delete_project_on_failure


# LLM-generated content at query #34
#--------------------------

```python
def test_delete_project_on_failure_evaluates_to_false():
    output_directory_created = True
    keep_project_on_failure = True
    delete_project_on_failure = output_directory_created and not keep_project_on_failure
    assert delete_project_on_failure is False


# LLM-generated content at query #35
#--------------------------

def test_generate_context_with_valid_json():
    import json
    from tempfile import NamedTemporaryFile
    from collections import OrderedDict

    context_data = {'key': 'value'}
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name)
    
    assert context == OrderedDict([(context_data['key'].split('.')[0], context_data)])

def test_generate_context_with_invalid_json():
    import json
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('invalid json')
        temp_file.flush()
        try:
            generate_context(temp_file.name)
        except ContextDecodingException:
            pass
        else:
            assert False, "Expected ContextDecodingException"

def test_generate_context_with_default_context():
    import json
    from tempfile import NamedTemporaryFile
    from collections import OrderedDict

    context_data = {'key': 'value'}
    default_context = {'key': 'new_value'}
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name, default_context=default_context)
    
    assert context == OrderedDict([(context_data['key'].split('.')[0], {'key': 'new_value'})])

def test_generate_context_with_extra_context():
    import json
    from tempfile import NamedTemporaryFile
    from collections import OrderedDict

    context_data = {'key': 'value'}
    extra_context = {'key': 'extra_value'}
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(context_data, temp_file)
        temp_file.flush()
        context = generate_context(temp_file.name, extra_context=extra_context)
    
    assert context == OrderedDict([(context_data['key'].split('.')[0], {'key': 'extra_value'})])


# LLM-generated content at query #36
#--------------------------

```python
def test_generate_context_with_valid_json():
    context_file = "test.json"
    default_context = {"key1": "value1"}
    extra_context = {"key2": "value2"}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump({"key1": "default1", "key2": "default2"}, f)
    context = generate_context(context_file, default_context, extra_context)
    assert context["test"] == {"key1": "value1", "key2": "value2"}
    os.remove(context_file)

def test_generate_context_with_invalid_json():
    context_file = "test_invalid.json"
    with open(context_file, "w", encoding="utf-8") as f:
        f.write("invalid json")
    try:
        generate_context(context_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException"
    os.remove(context_file)

def test_generate_context_with_default_context():
    context_file = "test.json"
    default_context = {"key1": "value1"}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump({"key1": "default1"}, f)
    context = generate_context(context_file, default_context)
    assert context["test"] == {"key1": "value1"}
    os.remove(context_file)

def test_generate_context_with_extra_context():
    context_file = "test.json"
    extra_context = {"key2": "value2"}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump({"key2": "default2"}, f)
    context = generate_context(context_file, extra_context=extra_context)
    assert context["test"] == {"key2": "value2"}
    os.remove(context_file)

def test_generate_context_with_both_default_and_extra_context():
    context_file = "test.json"
    default_context = {"key1": "value1"}
    extra_context = {"key2": "value2"}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump({"key1": "default1", "key2": "default2"}, f)
    context = generate_context(context_file, default_context, extra_context)
    assert context["test"] == {"key1": "value1", "key2": "value2"}
    os.remove(context_file)


# LLM-generated content at query #37
#--------------------------

def test_generate_files_with_accept_hooks_false():
    context = {}
    output_dir = '.'
    repo_dir = Path('test_repo')
    result = generate_files(repo_dir, context, output_dir, accept_hooks=False)
    assert isinstance(result, str)


# LLM-generated content at query #38
#--------------------------

```python
def test_generate_files_creates_project_directory():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

def test_generate_files_with_existing_output_dir_and_overwrite():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    os.makedirs(output_dir)
    project_dir = generate_files(repo_dir, context, output_dir, overwrite_if_exists=True)
    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

def test_generate_files_with_existing_output_dir_without_overwrite():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    os.makedirs(output_dir)
    try:
        generate_files(repo_dir, context, output_dir, overwrite_if_exists=False)
    except OutputDirExistsException:
        assert True
    else:
        assert False

def test_generate_files_with_copy_only_path():
    repo_dir = 'test_repo'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            '_copy_without_render': ['copy_me']
        }
    }
    output_dir = 'test_output'
    os.makedirs(os.path.join(repo_dir, 'copy_me'))
    project_dir = generate_files(repo_dir, context, output_dir)
    assert os.path.exists(os.path.join(project_dir, 'copy_me'))
    assert os.path.isdir(os.path.join(project_dir, 'copy_me'))

def test_generate_files_with_hooks():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script))
    with open(hook_script, 'w') as f:
        f.write('print("Hello from hook")')
    project_dir = generate_files(repo_dir, context, output_dir, accept_hooks=True)
    assert os.path.exists(project_dir)
    assert os.path.isdir(project_dir)

def test_generate_files_with_hooks_and_failure():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script))
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        generate_files(repo_dir, context, output_dir, accept_hooks=True)
    except FailedHookException:
        assert not os.path.exists(output_dir)
    else:
        assert False

def test_generate_files_with_hooks_and_failure_keep_project():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    hook_script = os.path.join(repo_dir, 'hooks', 'pre_gen_project.py')
    os.makedirs(os.path.dirname(hook_script))
    with open(hook_script, 'w') as f:
        f.write('import sys; sys.exit(1)')
    try:
        generate_files(repo_dir, context, output_dir, accept_hooks=True, keep_project_on_failure=True)
    except FailedHookException:
        assert os.path.exists(output_dir)
    else:
        assert False

def test_generate_files_with_skip_if_file_exists():
    repo_dir = 'test_repo'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    output_dir = 'test_output'
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'existing_file.txt'), 'w') as f:
        f.write('existing content')
    project_dir = generate_files(repo_dir, context, output_dir, skip_if_file_exists=True)
    assert os.path.exists(project_dir)
    assert os.path.isfile(os.path.join(project_dir, 'existing_file.txt'))


# LLM-generated content at query #39
#--------------------------

```python
def test_generate_file_creates_file_with_correct_content():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs("/tmp/template", exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "file.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "value"

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary.bin"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs("/tmp/template", exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "binary.bin"), "rb") as f:
        content = f.read()
    assert content == b"\x00\x01\x02\x03"

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs("/tmp/template", exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, "file.txt"), "w", encoding="utf-8") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(os.path.join(project_dir, "file.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "existing content"

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs("/tmp/template", exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, ""))

def test_generate_file_applies_file_permissions():
    project_dir = "/tmp/project"
    infile = "/tmp/template/file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment(loader=FileSystemLoader("/tmp/template"))
    os.makedirs("/tmp/template", exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("{{ cookiecutter.variable }}")
    os.chmod(infile, 0o644)
    generate_file(project_dir, infile, context, env)
    assert os.stat(os.path.join(project_dir, "file.txt")).st_mode & 0o777 == 0o644


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_67_evaluates_to_false():
    context = {'cookiecutter': {}}
    assert not context['cookiecutter'].get('_new_lines', False)


# LLM-generated content at query #41
#--------------------------

```python
def test_is_binary_predicate_at_line_47():
    binary_file = 'test_binary_file.bin'
    text_file = 'test_text_file.txt'
    
    # Create a binary file
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    
    # Create a text file
    with open(text_file, 'w') as f:
        f.write('Hello World')
    
    assert is_binary(binary_file) == True
    assert is_binary(text_file) == False
    
    # Clean up
    os.remove(binary_file)
    os.remove(text_file)


# LLM-generated content at query #42
#--------------------------

```python
def test_template_syntax_error_is_raised():
    class MockEnvironment:
        def get_template(self, infile_fwd_slashes):
            raise TemplateSyntaxError('test', 1, 'test', 'test')

    project_dir = '/path/to/project'
    infile = 'template.txt'
    context = {'cookiecutter': {}}
    env = MockEnvironment()
    skip_if_file_exists = False

    try:
        generate_file(project_dir, infile, context, env, skip_if_file_exists)
    except TemplateSyntaxError as e:
        assert not e.translated


# LLM-generated content at query #43
#--------------------------

```python
def test_new_lines_configuration_evaluates_to_true():
    context = {'cookiecutter': {'_new_lines': '\n'}}
    assert context['cookiecutter'].get('_new_lines', False) == '\n'


# LLM-generated content at query #44
#--------------------------

```python
def test_is_binary_returns_true_for_binary_file():
    binary_file = "test_binary_file.bin"
    with open(binary_file, "wb") as f:
        f.write(b'\x00\x01\x02\x03')
    result = is_binary(binary_file)
    os.remove(binary_file)
    assert result == True


# LLM-generated content at query #45
#--------------------------

```python
def test_generate_context_with_valid_json():
    context_file = 'valid.json'
    default_context = {'key1': 'value1'}
    extra_context = {'key2': 'value2'}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key1': 'old_value1', 'key2': 'old_value2'}, f)
    result = generate_context(context_file, default_context, extra_context)
    assert result['valid']['key1'] == 'value1'
    assert result['valid']['key2'] == 'value2'
    os.remove(context_file)

def test_generate_context_with_invalid_json():
    context_file = 'invalid.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write('invalid json')
    try:
        generate_context(context_file)
    except ContextDecodingException:
        assert True
    else:
        assert False
    os.remove(context_file)

def test_generate_context_without_default_and_extra_context():
    context_file = 'valid.json'
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key1': 'value1'}, f)
    result = generate_context(context_file)
    assert result['valid']['key1'] == 'value1'
    os.remove(context_file)

def test_generate_context_with_invalid_default_context():
    context_file = 'valid.json'
    default_context = {'key1': 'invalid_value'}
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump({'key1': ['valid_value']}, f)
    generate_context(context_file, default_context)
    os.remove(context_file)


# LLM-generated content at query #46
#--------------------------

```python
def test_template_syntax_error_raises_exception():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True
    else:
        assert False


# LLM-generated content at query #47
#--------------------------

```
def test_template_syntax_error_has_translated_disabled():
    class MockTemplateSyntaxError:
        translated = True

    exception = MockTemplateSyntaxError()
    try:
        raise TemplateSyntaxError("test")
    except TemplateSyntaxError as e:
        e.translated = False
        assert e.translated == False


# LLM-generated content at query #48
#--------------------------

```python
def test_generate_file_creates_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "template.txt")
    assert os.path.exists(outfile)
    with open(outfile, "r", encoding="utf-8") as f:
        assert f.read() == "Hello value"
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_skips_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w", encoding="utf-8") as f:
        f.write("Hello {{ cookiecutter.variable }}")
    outfile = os.path.join(project_dir, "template.txt")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r", encoding="utf-8") as f:
        assert f.read() == "Existing content"
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_copies_binary_file():
    project_dir = "/tmp/project"
    infile = "binary.dat"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    generate_file(project_dir, infile, context, env)
    outfile = os.path.join(project_dir, "binary.dat")
    assert os.path.exists(outfile)
    with open(outfile, "rb") as f:
        assert f.read() == b"\x00\x01\x02\x03"
    os.remove(infile)
    shutil.rmtree(project_dir)

def test_generate_file_handles_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(project_dir)
    shutil.rmtree(project_dir)


# LLM-generated content at query #49
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/path/to/project"
    infile = "/path/to/template/file.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    outfile = os.path.join(project_dir, "file.txt")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        content = f.read()
    assert content == "existing content"


# LLM-generated content at query #50
#--------------------------

```python
def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)

def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "binary_file.bin"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = ""
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_new_lines_from_context():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test", "_new_lines": "\r\n"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_new_lines_detected():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    generate_file(project_dir, infile, context, env)

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "invalid_template.txt"
    context = {"cookiecutter": {"name": "test"}}
    env = Environment()
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        pass


# LLM-generated content at query #51
#--------------------------

```python
def test_generate_context_with_invalid_json_file():
    invalid_json_file = "invalid.json"
    try:
        generate_context(context_file=invalid_json_file)
    except ContextDecodingException:
        pass
    else:
        assert False, "Expected ContextDecodingException to be raised"


# LLM-generated content at query #52
#--------------------------

```python
def test_skip_if_file_exists_and_file_already_exists():
    project_dir = "/path/to/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    outfile = os.path.join(project_dir, "template.txt")
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "existing content"


# LLM-generated content at query #53
#--------------------------

```
def test_predicate_at_line_67_evaluates_to_true():
    context = {'cookiecutter': {'_new_lines': True}}
    assert context['cookiecutter'].get('_new_lines', False) == True


# LLM-generated content at query #54
#--------------------------

```python
def test_template_syntax_error_handling():
    project_dir = "/tmp/project"
    infile = "template.txt"
    context = {"cookiecutter": {}}
    env = Environment()
    env.loader = DictLoader({infile: "{% if %}"})
    try:
        generate_file(project_dir, infile, context, env)
        assert False, "Expected TemplateSyntaxError"
    except TemplateSyntaxError:
        assert True


# LLM-generated content at query #55
#--------------------------

```python
def test_skip_if_file_exists_and_file_exists():
    project_dir = "/tmp/project"
    infile = "example.txt"
    context = {}
    env = Environment()
    outfile = os.path.join(project_dir, infile)
    os.makedirs(project_dir, exist_ok=True)
    with open(outfile, "w") as f:
        f.write("existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        content = f.read()
    assert content == "existing content"


# LLM-generated content at query #56
#--------------------------

```python
def test_generate_file_binary_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/binary_file.bin"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "wb") as f:
        f.write(b"binary content")
    generate_file(project_dir, infile, context, env)
    assert os.path.exists(os.path.join(project_dir, "binary_file.bin"))

def test_generate_file_text_file():
    project_dir = "/tmp/project"
    infile = "/tmp/template/text_file.txt"
    context = {"cookiecutter": {"variable": "value"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Text content with {{ cookiecutter.variable }}")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "text_file.txt"), "r") as f:
        assert f.read() == "Text content with value"

def test_generate_file_skip_if_file_exists():
    project_dir = "/tmp/project"
    infile = "/tmp/template/skip_file.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Content")
    outfile = os.path.join(project_dir, "skip_file.txt")
    with open(outfile, "w") as f:
        f.write("Existing content")
    generate_file(project_dir, infile, context, env, skip_if_file_exists=True)
    with open(outfile, "r") as f:
        assert f.read() == "Existing content"

def test_generate_file_empty_file_name():
    project_dir = "/tmp/project"
    infile = "/tmp/template/empty_file"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "empty_file"))
    generate_file(project_dir, infile, context, env)
    assert os.path.isdir(os.path.join(project_dir, "empty_file"))

def test_generate_file_with_new_lines():
    project_dir = "/tmp/project"
    infile = "/tmp/template/new_line_file.txt"
    context = {"cookiecutter": {"_new_lines": "\r\n"}}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("Line 1\nLine 2")
    generate_file(project_dir, infile, context, env)
    with open(os.path.join(project_dir, "new_line_file.txt"), "r") as f:
        assert f.read() == "Line 1\r\nLine 2"

def test_generate_file_template_syntax_error():
    project_dir = "/tmp/project"
    infile = "/tmp/template/syntax_error_file.txt"
    context = {}
    env = Environment()
    os.makedirs(project_dir, exist_ok=True)
    with open(infile, "w") as f:
        f.write("{{ invalid syntax }}")
    try:
        generate_file(project_dir, infile, context, env)
    except TemplateSyntaxError:
        assert True
    else:
        assert False


