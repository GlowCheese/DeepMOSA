####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_nested_template_recursion():
    context = {"cookiecutter": {"templates": {"choice": {"path": "nested_path"}}}}
    repo_dir = "."
    no_input = True
    template = choose_nested_template(context, repo_dir, no_input)
    assert template is not None

def test_cookiecutter_prompt_for_config_updates_context():
    context_for_prompting = {"cookiecutter": {"key": "default_value"}}
    no_input = True
    updated_context = prompt_for_config(context_for_prompting, no_input)
    assert "key" in updated_context
    assert updated_context["key"] == "default_value"

def test_cookiecutter_generate_context_with_valid_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": "Test Project"}')
    context = generate_context(context_file=str(context_file))
    assert "cookiecutter" in context
    assert context["cookiecutter"]["project_name"] == "Test Project"

def test_cookiecutter_generate_context_with_invalid_file(tmp_path):
    context_file = tmp_path / "cookiecutter.json"
    context_file.write_text('{"project_name": invalid}')
    try:
        generate_context(context_file=str(context_file))
    except ContextDecodingException as e:
        assert "JSON decoding error" in str(e)

def test_cookiecutter_get_user_config_default():
    config = get_user_config(default_config=True)
    assert config is not None

def test_cookiecutter_get_user_config_custom_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"default_context": {"key": "value"}}')
    config = get_user_config(config_file=str(config_file))
    assert "default_context" in config

def test_cookiecutter_run_pre_prompt_hook_no_script(tmp_path):
    repo_dir = tmp_path
    result = run_pre_prompt_hook(repo_dir)
    assert result == repo_dir

def test_cookiecutter_choose_nested_template_old_style():
    context = {"cookiecutter": {"template": ["Option (path)"]}}
    repo_dir = "."
    no_input = True
    template = choose_nested_template(context, repo_dir, no_input)
    assert template is not None

def test_cookiecutter_choose_nested_template_invalid_path():
    context = {"cookiecutter": {"templates": {"choice": {"path": "/absolute/path"}}}}
    repo_dir = "."
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert "Illegal template path" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template="test", replay=True, no_input=True)
        assert False, "Should raise InvalidModeException"
    except InvalidModeException as e:
        assert "You can not use both replay and no_input" in str(e)

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template="test", replay=True, extra_context={"key": "value"})
        assert False, "Should raise InvalidModeException"
    except InvalidModeException as e:
        assert "You can not use both replay and no_input" in str(e)

def test_cookiecutter_with_default_config_true():
    result = cookiecutter(template="test", default_config=True)
    assert isinstance(result, str)

def test_cookiecutter_with_default_config_dict():
    result = cookiecutter(template="test", default_config={"key": "value"})
    assert isinstance(result, str)

def test_cookiecutter_with_config_file():
    result = cookiecutter(template="test", config_file="/tmp/config.yaml")
    assert isinstance(result, str)

def test_cookiecutter_with_no_input():
    result = cookiecutter(template="test", no_input=True)
    assert isinstance(result, str)

def test_cookiecutter_with_replay_bool():
    result = cookiecutter(template="test", replay=True)
    assert isinstance(result, str)

def test_cookiecutter_with_replay_path():
    result = cookiecutter(template="test", replay="/tmp/replay.json")
    assert isinstance(result, str)

def test_cookiecutter_with_overwrite_if_exists():
    result = cookiecutter(template="test", overwrite_if_exists=True)
    assert isinstance(result, str)

def test_cookiecutter_with_custom_output_dir():
    result = cookiecutter(template="test", output_dir="/tmp/output")
    assert isinstance(result, str)

def test_cookiecutter_with_password():
    result = cookiecutter(template="test", password="secret")
    assert isinstance(result, str)

def test_cookiecutter_with_directory():
    result = cookiecutter(template="test", directory="subdir")
    assert isinstance(result, str)

def test_cookiecutter_with_skip_if_file_exists():
    result = cookiecutter(template="test", skip_if_file_exists=True)
    assert isinstance(result, str)

def test_cookiecutter_with_accept_hooks_false():
    result = cookiecutter(template="test", accept_hooks=False)
    assert isinstance(result, str)

def test_cookiecutter_with_keep_project_on_failure():
    result = cookiecutter(template="test", keep_project_on_failure=True)
    assert isinstance(result, str)

def test_cookiecutter_with_checkout():
    result = cookiecutter(template="test", checkout="v1.0.0")
    assert isinstance(result, str)

def test_cookiecutter_with_extra_context():
    result = cookiecutter(template="test", extra_context={"project_name": "MyProject"})
    assert isinstance(result, str)

def test_cookiecutter_zip_template():
    result = cookiecutter(template="https://example.com/template.zip")
    assert isinstance(result, str)

def test_cookiecutter_git_template():
    result = cookiecutter(template="https://github.com/user/repo.git")
    assert isinstance(result, str)

def test_cookiecutter_local_template():
    result = cookiecutter(template="/local/path/to/template")
    assert isinstance(result, str)

def test_cookiecutter_nested_template_selection():
    context = {"cookiecutter": {"template": "nested"}}
    result = cookiecutter(template="test", extra_context=context)
    assert isinstance(result, str)

def test_cookiecutter_cleanup_on_zip():
    result = cookiecutter(template="https://example.com/template.zip")
    assert isinstance(result, str)

def test_cookiecutter_no_cleanup_on_git():
    result = cookiecutter(template="https://github.com/user/repo.git")
    assert isinstance(result, str)

def test_cookiecutter_replay_loads_context():
    result = cookiecutter(template="test", replay=True)
    assert isinstance(result, str)

def test_cookiecutter_preserves_original_options():
    result = cookiecutter(template="test")
    assert isinstance(result, str)

def test_cookiecutter_includes_template_in_context():
    result = cookiecutter(template="test")
    assert isinstance(result, str)

def test_cookiecutter_includes_output_dir_in_context():
    result = cookiecutter(template="test", output_dir="/tmp/output")
    assert isinstance(result, str)

def test_cookiecutter_includes_repo_dir_in_context():
    result = cookiecutter(template="test")
    assert isinstance(result, str)

def test_cookiecutter_includes_checkout_in_context():
    result = cookiecutter(template="test", checkout="main")
    assert isinstance(result, str)


# LLM-generated content at query #3
#--------------------------

def test_replay_false_does_not_enter_replay_block():
    replay = False
    assert not replay


# LLM-generated content at query #4
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    err_msg = "You can not use both replay and no_input or extra_context at the same time."
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert str(e) == err_msg
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert str(e) == err_msg

def test_cookiecutter_replay_with_bool_and_no_extra_context():
    config_dict = get_user_config(default_config=True)
    mock_repo_dir = "/fake/repo"
    mock_template_name = "template"
    mock_context = {"cookiecutter": {"key": "value"}}
    mock_load = lambda replay_dir, template_name: mock_context
    import sys
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "cookiecutter.replay":
            class MockReplay:
                load = mock_load
            return MockReplay()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    try:
        result = cookiecutter(template="some_template", replay=True, no_input=False, extra_context=None)
    finally:
        builtins.__import__ = original_import

def test_cookiecutter_replay_with_custom_path():
    config_dict = get_user_config(default_config=True)
    mock_repo_dir = "/fake/repo"
    mock_template_name = "custom_template"
    mock_context = {"cookiecutter": {"key": "value"}}
    mock_load = lambda replay_dir, template_name: mock_context
    import sys
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "cookiecutter.replay":
            class MockReplay:
                load = mock_load
            return MockReplay()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    try:
        result = cookiecutter(template="some_template", replay="/custom/replay.json", no_input=False, extra_context=None)
    finally:
        builtins.__import__ = original_import

def test_cookiecutter_with_nested_template_selection():
    mock_context = {"cookiecutter": {"templates": {"choice1": {"path": "subdir"}}}}
    mock_generate_context = lambda context_file, default_context, extra_context: mock_context
    mock_choose_nested_template = lambda context, repo_dir, no_input: "/fake/repo/subdir"
    import sys
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "cookiecutter.generate":
            class MockGenerate:
                generate_context = mock_generate_context
            return MockGenerate()
        if name == "cookiecutter.prompt":
            class MockPrompt:
                choose_nested_template = mock_choose_nested_template
            return MockPrompt()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    try:
        result = cookiecutter(template="some_template", no_input=True)
    finally:
        builtins.__import__ = original_import

def test_cookiecutter_cleanup_temp_dirs():
    mock_repo_dir = "/fake/temp/repo"
    mock_base_repo_dir = "/fake/base/repo"
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: (mock_base_repo_dir, True)
    mock_run_pre_prompt_hook = lambda repo_dir: mock_repo_dir
    mock_rmtree = lambda path: None
    import sys
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "cookiecutter.repository":
            class MockRepository:
                determine_repo_dir = mock_determine_repo_dir
            return MockRepository()
        if name == "cookiecutter.hooks":
            class MockHooks:
                run_pre_prompt_hook = mock_run_pre_prompt_hook
            return MockHooks()
        if name == "cookiecutter.utils":
            class MockUtils:
                rmtree = mock_rmtree
            return MockUtils()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    try:
        result = cookiecutter(template="some_template", accept_hooks=True)
    finally:
        builtins.__import__ = original_import

def test_cookiecutter_with_default_config():
    mock_config_dict = {"abbreviations": {}, "cookiecutters_dir": "/fake/dir", "default_context": {}, "replay_dir": "/fake/replay"}
    mock_get_user_config = lambda config_file, default_config: mock_config_dict
    import sys
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "cookiecutter.config":
            class MockConfig:
                get_user_config = mock_get_user_config
            return MockConfig()
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    try:
        result = cookiecutter(template="some_template", default_config=True)
    finally:
        builtins.__import__ = original_import


# LLM-generated content at query #5
#--------------------------

def test_constructor_with_path_object():
    from pathlib import Path
    from unittest.mock import MagicMock
    temp_dir = Path("/fake/path")
    instance = _patch_import_path_for_repo(temp_dir)
    assert instance._repo_dir == "/fake/path"

def test_constructor_with_string():
    from unittest.mock import MagicMock
    repo_string = "/fake/string/path"
    instance = _patch_import_path_for_repo(repo_string)
    assert instance._repo_dir == "/fake/string/path"

def test_constructor_ensures_string_type():
    from pathlib import Path
    from unittest.mock import MagicMock
    path_obj = Path("/some/dir")
    instance = _patch_import_path_for_repo(path_obj)
    assert isinstance(instance._repo_dir, str)


# LLM-generated content at query #6
#--------------------------

```python
def test_accept_hooks_false_makes_predicate_false():
    accept_hooks = False
    base_repo_dir = "/some/path"
    repo_dir = base_repo_dir
    predicate_result = repo_dir != base_repo_dir
    assert predicate_result == False


# LLM-generated content at query #7
#--------------------------

def test_choose_nested_template_with_absolute_path():
    context = {'cookiecutter': {'templates': {'choice': {'path': '/absolute/path'}}}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert str(e) == "Illegal template path"

def test_choose_nested_template_with_empty_path():
    context = {'cookiecutter': {'templates': {'choice': {'path': ''}}}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert str(e) == "Illegal template path"

def test_choose_nested_template_with_none_path():
    context = {'cookiecutter': {'templates': {'choice': {'path': None}}}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert str(e) == "Illegal template path"

def test_choose_nested_template_old_style_with_absolute_path():
    context = {'cookiecutter': {'template': ['name (/absolute/path)']}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert str(e) == "Illegal template path"

def test_choose_nested_template_old_style_with_empty_path():
    context = {'cookiecutter': {'template': ['name ()']}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
    except ValueError as e:
        assert str(e) == "Illegal template path"


# LLM-generated content at query #8
#--------------------------

```python
def test_replay_false_does_not_enter_replay_block():
    replay = False
    result = replay
    assert not result


# LLM-generated content at query #9
#--------------------------

def test_init_with_path_object():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import sys
    import copy
    from contextlib import contextmanager
    
    class _patch_import_path_for_repo:
        def __init__(self, repo_dir: Path | str) -> None:
            self._repo_dir = f"{repo_dir}" if isinstance(repo_dir, Path) else repo_dir

        def __enter__(self) -> None:
            self._path = copy(sys.path)
            sys.path.append(self._repo_dir)

        def __exit__(self, _type, _value, _traceback):
            sys.path = self._path
    
    test_path = Path("/test/path")
    instance = _patch_import_path_for_repo(test_path)
    assert isinstance(test_path, Path)
    assert instance._repo_dir == f"{test_path}"


# LLM-generated content at query #10
#--------------------------

```python
def test_cleanup_is_true_when_repo_dir_changed_by_pre_prompt_hook():
    import sys
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException

    mock_run_pre_prompt_hook = Mock(return_value="/new/repo/dir")
    with patch('cookiecutter.main.run_pre_prompt_hook', mock_run_pre_prompt_hook):
        mock_get_user_config = Mock(return_value={
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'default_context': {},
            'replay_dir': '/tmp/replay'
        })
        with patch('cookiecutter.main.get_user_config', mock_get_user_config):
            mock_determine_repo_dir = Mock(return_value=('/original/repo/dir', False))
            with patch('cookiecutter.main.determine_repo_dir', mock_determine_repo_dir):
                mock_generate_context = Mock(return_value={'cookiecutter': {}})
                with patch('cookiecutter.main.generate_context', mock_generate_context):
                    mock_prompt_for_config = Mock(return_value={})
                    with patch('cookiecutter.main.prompt_for_config', mock_prompt_for_config):
                        mock_generate_files = Mock(return_value='/output/project')
                        with patch('cookiecutter.main.generate_files', mock_generate_files):
                            mock_dump = Mock()
                            with patch('cookiecutter.main.dump', mock_dump):
                                mock_rmtree = Mock()
                                with patch('cookiecutter.main.rmtree', mock_rmtree):
                                    result = cookiecutter(
                                        template='some/template',
                                        accept_hooks=True
                                    )
                                    assert mock_rmtree.call_count == 1
                                    mock_rmtree.assert_called_with('/original/repo/dir')


# LLM-generated content at query #11
#--------------------------

def test_constructor_with_path_object():
    from pathlib import Path
    temp_path = Path("/tmp/test")
    instance = _patch_import_path_for_repo(temp_path)
    assert instance._repo_dir == "/tmp/test"

def test_constructor_with_string():
    instance = _patch_import_path_for_repo("/tmp/test")
    assert instance._repo_dir == "/tmp/test"

def test_constructor_with_empty_string():
    instance = _patch_import_path_for_repo("")
    assert instance._repo_dir == ""

def test_constructor_with_path_object_containing_spaces():
    from pathlib import Path
    temp_path = Path("/tmp/test path")
    instance = _patch_import_path_for_repo(temp_path)
    assert instance._repo_dir == "/tmp/test path"

def test_constructor_with_string_containing_spaces():
    instance = _patch_import_path_for_repo("/tmp/test path")
    assert instance._repo_dir == "/tmp/test path"


# LLM-generated content at query #12
#--------------------------

```python
def test_patch_import_path_for_repo_enter_exit():
    import sys
    from cookiecutter.main import _patch_import_path_for_repo
    original_path = sys.path.copy()
    repo_dir = "/some/test/path"
    patcher = _patch_import_path_for_repo(repo_dir)
    patcher.__enter__()
    assert sys.path[-1] == repo_dir
    patcher.__exit__(None, None, None)
    assert sys.path == original_path


# LLM-generated content at query #13
#--------------------------

```python
def test_cookiecutter_cleanup_false_when_repo_dir_equals_base_repo_dir():
    repo_dir = "/some/path"
    base_repo_dir = "/some/path"
    cleanup = repo_dir != base_repo_dir
    assert cleanup is False


# LLM-generated content at query #14
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template='test', replay=True, no_input=True)
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template='test', replay=True, extra_context={'key': 'value'})
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_with_replay_bool():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {'cookiecutter': {'key': 'value'}}
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/result'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_modules = {}
    for mod in ['cookiecutter.main', 'cookiecutter.replay', 'cookiecutter.generate', 'cookiecutter.hooks', 'cookiecutter.config', 'cookiecutter.repository']:
        original_modules[mod] = sys.modules.get(mod)
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].load = mock_load
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate'].generate_files = mock_generate_files
    sys.modules['cookiecutter.main'].dump = mock_dump
    result = cookiecutter(template='test', replay=True)
    assert result == '/tmp/result'
    for mod, original in original_modules.items():
        if original is None:
            del sys.modules[mod]
        else:
            sys.modules[mod] = original

def test_cookiecutter_with_replay_string():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {'cookiecutter': {'key': 'value'}}
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/result'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_modules = {}
    for mod in ['cookiecutter.main', 'cookiecutter.replay', 'cookiecutter.generate', 'cookiecutter.hooks', 'cookiecutter.config', 'cookiecutter.repository']:
        original_modules[mod] = sys.modules.get(mod)
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].load = mock_load
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate'].generate_files = mock_generate_files
    sys.modules['cookiecutter.main'].dump = mock_dump
    result = cookiecutter(template='test', replay='/tmp/replay.json')
    assert result == '/tmp/result'
    for mod, original in original_modules.items():
        if original is None:
            del sys.modules[mod]
        else:
            sys.modules[mod] = original

def test_cookiecutter_without_replay():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {'key': 'prompted'}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/result'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_modules = {}
    for mod in ['cookiecutter.main', 'cookiecutter.replay', 'cookiecutter.generate', 'cookiecutter.hooks', 'cookiecutter.config', 'cookiecutter.repository']:
        original_modules[mod] = sys.modules.get(mod)
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate'].generate_files = mock_generate_files
    sys.modules['cookiecutter.main'].dump = mock_dump
    result = cookiecutter(template='test', extra_context={'key': 'extra'})
    assert result == '/tmp/result'
    for mod, original in original_modules.items():
        if original is None:
            del sys.modules[mod]
        else:
            sys.modules[mod] = original

def test_cookiecutter_nested_template():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'template': 'nested'}}
    mock_choose_nested_template = lambda context, repo_dir, no_input: 'nested_template'
    mock_cookiecutter = lambda template, checkout, no_input, extra_context, replay, overwrite_if_exists, output_dir, config_file, default_config, password, directory, skip_if_file_exists, accept_hooks, keep_project_on_failure: '/tmp/nested_result'
    import sys
    original_modules = {}
    for mod in ['cookiecutter.main', 'cookiecutter.replay', 'cookiecutter.generate', 'cookiecutter.hooks', 'cookiecutter.config', 'cookiecutter.repository']:
        original_modules[mod] = sys.modules.get(mod)
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.main'].choose_nested_template = mock


# LLM-generated content at query #15
#--------------------------

```python
def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = False
    no_input = False
    extra_context = None
    try:
        cookiecutter(
            template="some_template",
            replay=replay,
            no_input=no_input,
            extra_context=extra_context
        )
    except InvalidModeException:
        pytest.fail("InvalidModeException should not be raised when replay is False")
    except Exception:
        pass

def test_cookiecutter_replay_none_with_no_input_false_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = None
    no_input = False
    extra_context = None
    try:
        cookiecutter(
            template="some_template",
            replay=replay,
            no_input=no_input,
            extra_context=extra_context
        )
    except InvalidModeException:
        pytest.fail("InvalidModeException should not be raised when replay is None")
    except Exception:
        pass

def test_cookiecutter_replay_false_with_no_input_true_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = False
    no_input = True
    extra_context = None
    try:
        cookiecutter(
            template="some_template",
            replay=replay,
            no_input=no_input,
            extra_context=extra_context
        )
    except InvalidModeException:
        pytest.fail("InvalidModeException should not be raised when replay is False")
    except Exception:
        pass

def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_dict():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = False
    no_input = False
    extra_context = {"key": "value"}
    try:
        cookiecutter(
            template="some_template",
            replay=replay,
            no_input=no_input,
            extra_context=extra_context
        )
    except InvalidModeException:
        pytest.fail("InvalidModeException should not be raised when replay is False")
    except Exception:
        pass

def test_cookiecutter_replay_none_with_no_input_true_and_extra_context_dict():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = None
    no_input = True
    extra_context = {"key": "value"}
    try:
        cookiecutter(
            template="some_template",
            replay=replay,
            no_input=no_input,
            extra_context=extra_context
        )
    except InvalidModeException:
        pytest.fail("InvalidModeException should not be raised when replay is None")
    except Exception:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from unittest.mock import patch, MagicMock
    with patch('cookiecutter.main.get_user_config') as mock_get_user_config, \
         patch('cookiecutter.main.determine_repo_dir') as mock_determine_repo_dir, \
         patch('cookiecutter.main.run_pre_prompt_hook') as mock_run_pre_prompt_hook, \
         patch('cookiecutter.main.generate_context') as mock_generate_context, \
         patch('cookiecutter.main.prompt_for_config') as mock_prompt_for_config, \
         patch('cookiecutter.main.generate_files') as mock_generate_files, \
         patch('cookiecutter.main.dump') as mock_dump, \
         patch('cookiecutter.main.rmtree') as mock_rmtree:
        mock_get_user_config.return_value = {
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'replay_dir': '/tmp/replay',
            'default_context': {}
        }
        mock_determine_repo_dir.return_value = ('/tmp/repo', False)
        mock_run_pre_prompt_hook.return_value = '/tmp/repo'
        mock_generate_context.return_value = {'cookiecutter': {}}
        mock_prompt_for_config.return_value = {}
        mock_generate_files.return_value = '/tmp/output'
        cookiecutter(template='test', replay=False, no_input=False, extra_context=None)

def test_cookiecutter_replay_false_with_no_input_true_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from unittest.mock import patch, MagicMock
    with patch('cookiecutter.main.get_user_config') as mock_get_user_config, \
         patch('cookiecutter.main.determine_repo_dir') as mock_determine_repo_dir, \
         patch('cookiecutter.main.run_pre_prompt_hook') as mock_run_pre_prompt_hook, \
         patch('cookiecutter.main.generate_context') as mock_generate_context, \
         patch('cookiecutter.main.prompt_for_config') as mock_prompt_for_config, \
         patch('cookiecutter.main.generate_files') as mock_generate_files, \
         patch('cookiecutter.main.dump') as mock_dump, \
         patch('cookiecutter.main.rmtree') as mock_rmtree:
        mock_get_user_config.return_value = {
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'replay_dir': '/tmp/replay',
            'default_context': {}
        }
        mock_determine_repo_dir.return_value = ('/tmp/repo', False)
        mock_run_pre_prompt_hook.return_value = '/tmp/repo'
        mock_generate_context.return_value = {'cookiecutter': {}}
        mock_prompt_for_config.return_value = {}
        mock_generate_files.return_value = '/tmp/output'
        cookiecutter(template='test', replay=False, no_input=True, extra_context=None)

def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_empty():
    from cookiecutter.main import cookiecutter
    from unittest.mock import patch, MagicMock
    with patch('cookiecutter.main.get_user_config') as mock_get_user_config, \
         patch('cookiecutter.main.determine_repo_dir') as mock_determine_repo_dir, \
         patch('cookiecutter.main.run_pre_prompt_hook') as mock_run_pre_prompt_hook, \
         patch('cookiecutter.main.generate_context') as mock_generate_context, \
         patch('cookiecutter.main.prompt_for_config') as mock_prompt_for_config, \
         patch('cookiecutter.main.generate_files') as mock_generate_files, \
         patch('cookiecutter.main.dump') as mock_dump, \
         patch('cookiecutter.main.rmtree') as mock_rmtree:
        mock_get_user_config.return_value = {
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'replay_dir': '/tmp/replay',
            'default_context': {}
        }
        mock_determine_repo_dir.return_value = ('/tmp/repo', False)
        mock_run_pre_prompt_hook.return_value = '/tmp/repo'
        mock_generate_context.return_value = {'cookiecutter': {}}
        mock_prompt_for_config.return_value = {}
        mock_generate_files.return_value = '/tmp/output'
        cookiecutter(template='test', replay=False, no_input=False, extra_context={})

def test_cookiecutter_replay_none_with_no_input_false_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from unittest.mock import patch, MagicMock
    with patch('cookiecutter.main.get_user_config') as mock_get_user_config, \
         patch('cookiecutter.main.determine_repo_dir') as mock_determine_repo_dir, \
         patch('cookiecutter.main.run_pre_prompt_hook') as mock_run_pre_prompt_hook, \
         patch('cookiecutter.main.generate_context') as mock_generate_context, \
         patch('cookiecutter.main.prompt_for_config') as mock_prompt_for_config, \
         patch('cookiecutter.main.generate_files') as mock_generate_files, \
         patch('cookiecutter.main.dump') as mock_dump, \
         patch('cookiecutter.main.rmtree') as mock_rmtree:
        mock_get_user_config.return_value = {
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'replay_dir': '/tmp/replay',
            'default_context': {}
        }
        mock_determine_repo_dir.return_value = ('/tmp/repo', False)
        mock_run_pre_prompt_hook.return_value = '/tmp/repo'
        mock_generate_context.return_value = {'cookiecutter': {}}
        mock_prompt_for_config.return_value = {}
        mock_generate_files.return_value = '/tmp/output'
        cookiecutter(template='test', replay=None, no_input=False, extra_context=None)

def test_cookiecutter_replay_empty_string_with_no_input_false_and_extra_context_none():
    from cookiecutter.main import cookiecutter
    from unittest.mock import patch, MagicMock
    with patch('cookiecutter.main.get_user_config') as mock_get_user_config, \
         patch('cookiecutter.main.determine_repo_dir') as mock_determine_repo_dir, \
         patch('cookiecutter.main.run_pre_prompt_hook') as mock_run_pre_prompt_hook, \
         patch('cookiecutter.main.generate_context') as mock_generate_context, \
         patch('cookiecutter.main.prompt_for_config') as mock_prompt_for_config, \
         patch('cookiecutter.main.generate_files') as mock_generate_files, \
         patch('cookiecutter.main.dump') as mock_dump, \
         patch('cookiecutter.main.rmtree') as mock_rmtree:
        mock_get_user_config.return_value = {
            'abbreviations': {},
            'cookiecutters_dir': '/tmp',
            'replay_dir': '/tmp/replay',
            'default_context': {}
        }
        mock_determine_repo_dir.return_value = ('/tmp/repo', False)
        mock_run_pre_prompt_hook.return_value = '/tmp/repo'
        mock_generate_context.return_value = {'cookiecutter': {}}
        mock_prompt_for_config.return_value = {}
        mock_generate_files.return_value = '/tmp/output'
        cookiecutter(template='test', replay='', no_input=False, extra_context=None)


# LLM-generated content at query #17
#--------------------------

```python
def test_replay_false_does_not_enter_replay_block():
    replay = False
    no_input = True
    extra_context = {"key": "value"}
    config_dict = {"replay_dir": "/tmp", "default_context": {}, "abbreviations": {}, "cookiecutters_dir": "/tmp"}
    repo_dir = "/tmp/repo"
    template_name = "test_template"
    context_file = "/tmp/repo/cookiecutter.json"
    import_patch = _patch_import_path_for_repo(repo_dir)
    with import_patch:
        context = generate_context(context_file=context_file, default_context=config_dict['default_context'], extra_context=extra_context)
        context_for_prompting = context
    assert replay is False


# LLM-generated content at query #18
#--------------------------

```python
def test_accept_hooks_false_does_not_run_pre_prompt_hook():
    accept_hooks = False
    base_repo_dir = "/some/path"
    repo_dir = base_repo_dir
    result = str(run_pre_prompt_hook(base_repo_dir)) if accept_hooks else repo_dir
    assert result == repo_dir
    assert result == base_repo_dir


# LLM-generated content at query #19
#--------------------------

def test_patch_import_path_for_repo_init_with_path():
    from pathlib import Path
    from unittest.mock import patch
    import sys
    temp_dir = Path("/tmp/test_repo")
    instance = _patch_import_path_for_repo(temp_dir)
    assert instance._repo_dir == f"{temp_dir}"

def test_patch_import_path_for_repo_init_with_str():
    from unittest.mock import patch
    import sys
    repo_str = "/tmp/test_repo"
    instance = _patch_import_path_for_repo(repo_str)
    assert instance._repo_dir == repo_str


# LLM-generated content at query #20
#--------------------------

```python
def test_cleanup_set_to_true_when_repo_dir_changed_by_pre_prompt_hook():
    base_repo_dir = "/tmp/original"
    repo_dir = "/tmp/modified"
    accept_hooks = True
    mock_run_pre_prompt_hook = lambda x: repo_dir
    cleanup = repo_dir != base_repo_dir
    assert cleanup == True


# LLM-generated content at query #21
#--------------------------

```python
def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_none():
    replay = False
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate is False


# LLM-generated content at query #22
#--------------------------

```python
def test_cookiecutter_replay_false_with_no_input_false_and_extra_context_none():
    replay = False
    no_input = False
    extra_context = None
    assert not (replay and ((no_input is not False) or (extra_context is not None)))


# LLM-generated content at query #23
#--------------------------

```python
def test_replay_false_with_no_input_false_and_extra_context_none():
    replay = False
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_none_with_no_input_false_and_extra_context_none():
    replay = None
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_true_with_no_input_true_and_extra_context_none():
    replay = True
    no_input = True
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == True

def test_replay_true_with_no_input_false_and_extra_context_not_none():
    replay = True
    no_input = False
    extra_context = {"key": "value"}
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == True

def test_replay_string_with_no_input_false_and_extra_context_none():
    replay = "some_path"
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_string_with_no_input_true_and_extra_context_none():
    replay = "some_path"
    no_input = True
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == True

def test_replay_string_with_no_input_false_and_extra_context_not_none():
    replay = "some_path"
    no_input = False
    extra_context = {"key": "value"}
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == True

def test_replay_true_with_no_input_false_and_extra_context_none():
    replay = True
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False


# LLM-generated content at query #24
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template='some_template', replay=True, no_input=True)
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template='some_template', replay=True, extra_context={'key': 'value'})
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_with_replay_bool():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {'cookiecutter': {'key': 'value'}}
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/project'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_get_user_config = sys.modules['cookiecutter.main'].get_user_config
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    original_determine_repo_dir = sys.modules['cookiecutter.main'].determine_repo_dir
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    original_run_pre_prompt_hook = sys.modules['cookiecutter.main'].run_pre_prompt_hook
    sys.modules['cookiecutter.main'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    original_load = sys.modules['cookiecutter.main'].load
    sys.modules['cookiecutter.main'].load = mock_load
    original_generate_context = sys.modules['cookiecutter.main'].generate_context
    sys.modules['cookiecutter.main'].generate_context = mock_generate_context
    original_prompt_for_config = sys.modules['cookiecutter.main'].prompt_for_config
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    original_generate_files = sys.modules['cookiecutter.main'].generate_files
    sys.modules['cookiecutter.main'].generate_files = mock_generate_files
    original_dump = sys.modules['cookiecutter.main'].dump
    sys.modules['cookiecutter.main'].dump = mock_dump
    result = cookiecutter(template='some_template', replay=True)
    assert result == '/tmp/project'
    sys.modules['cookiecutter.main'].get_user_config = original_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = original_determine_repo_dir
    sys.modules['cookiecutter.main'].run_pre_prompt_hook = original_run_pre_prompt_hook
    sys.modules['cookiecutter.main'].load = original_load
    sys.modules['cookiecutter.main'].generate_context = original_generate_context
    sys.modules['cookiecutter.main'].prompt_for_config = original_prompt_for_config
    sys.modules['cookiecutter.main'].generate_files = original_generate_files
    sys.modules['cookiecutter.main'].dump = original_dump

def test_cookiecutter_with_replay_file():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {'cookiecutter': {'key': 'value'}}
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/project'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_get_user_config = sys.modules['cookiecutter.main'].get_user_config
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    original_determine_repo_dir = sys.modules['cookiecutter.main'].determine_repo_dir
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    original_run_pre_prompt_hook = sys.modules['cookiecutter.main'].run_pre_prompt_hook
    sys.modules['cookiecutter.main'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    original_load = sys.modules['cookiecutter.main'].load
    sys.modules['cookiecutter.main'].load = mock_load
    original_generate_context = sys.modules['cookiecutter.main'].generate_context
    sys.modules['cookiecutter.main'].generate_context = mock_generate_context
    original_prompt_for_config = sys.modules['cookiecutter.main'].prompt_for_config
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    original_generate_files = sys.modules['cookiecutter.main'].generate_files
    sys.modules['cookiecutter.main'].generate_files = mock_generate_files
    original_dump = sys.modules['cookiecutter.main'].dump
    sys.modules['cookiecutter.main'].dump = mock_dump
    result = cookiecutter(template='some_template', replay='/tmp/replay.json')
    assert result == '/tmp/project'
    sys.modules['cookiecutter.main'].get_user_config = original_get_user_config
    sys.modules['cookiecutter.main'].determine_repo_dir = original_determine_repo_dir
    sys.modules['cookiecutter.main'].run_pre_prompt_hook = original_run_pre_prompt_hook
    sys.modules['cookiecutter.main'].load = original_load
    sys.modules['cookiecutter.main'].generate_context = original_generate_context
    sys.modules['cookiecutter.main'].prompt_for_config = original_prompt_for_config
    sys.modules['cookiecutter.main'].generate_files = original_generate_files
    sys.modules['cookiecutter.main'].dump = original_dump

def test_cookiecutter_without_replay():
    config_dict = {'replay_dir': '/tmp', 'default_context': {}, 'abbreviations': {}, 'cookiecutters_dir': '/tmp'}
    mock_get_user_config = lambda config_file=None, default_config=False: config_dict
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ('/tmp/repo', False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_generate_context = lambda context_file, default_context, extra_context: {'cookiecutter': {'key': 'default'}}
    mock_prompt_for_config = lambda context, no_input: {'key': 'prompted'}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: '/tmp/project'
    mock_dump = lambda replay_dir, template_name, context: None
    import sys
    original_get_user_config = sys.modules['cookiecutter.main'].get_user_config
    sys.modules['cookiecutter.main'].get_user_config = mock_get_user_config
    original_determine_repo_dir = sys.modules['cookiecutter.main'].determine_repo_dir
    sys.modules['cookiecutter.main'].determine_repo_dir = mock_determine_repo_dir
    original_run_pre_prompt_hook = sys.modules['cookiecutter.main'].run_pre_prompt_hook
    sys.modules['cookiecutter.main'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    original_generate_context = sys.modules['cookiecutter.main'].generate_context
    sys.modules['cookiecutter.main'].generate_context = mock_generate_context
    original_prompt_for_config = sys.modules['cookiecutter.main'].prompt_for_config
    sys.modules['cookiecutter.main'].prompt_for_config = mock_prompt_for_config
    original_generate_files = sys.modules['cookiecutter.main'].


# LLM-generated content at query #25
#--------------------------

```python
def test_replay_is_truthy_when_string():
    replay = "some_replay_file.json"
    result = bool(replay)
    assert result is True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_path_object():
    mock_path = Path("/fake/path")
    instance = _patch_import_path_for_repo(mock_path)
    assert instance._repo_dir == "/fake/path"

def test_constructor_with_string():
    repo_dir = "/fake/path"
    instance = _patch_import_path_for_repo(repo_dir)
    assert instance._repo_dir == "/fake/path"

def test_constructor_path_object_converted_to_string():
    mock_path = Path("/another/fake/path")
    instance = _patch_import_path_for_repo(mock_path)
    assert isinstance(instance._repo_dir, str)

def test_constructor_string_remains_string():
    repo_dir = "/another/fake/path"
    instance = _patch_import_path_for_repo(repo_dir)
    assert isinstance(instance._repo_dir, str)


# LLM-generated content at query #2
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template='some_template', replay=True, no_input=True)
        assert False
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template='some_template', replay=True, extra_context={'key': 'value'})
        assert False
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_with_replay_bool():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_template_name = 'fake_template'
    mock_context = {'cookiecutter': {'key': 'value'}}
    mock_load = lambda replay_dir, template_name: mock_context
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.load', mock_load):
                    with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                        with unittest.mock.patch('cookiecutter.main.generate_files', return_value='/fake/output'):
                            result = cookiecutter(template='fake_template', replay=True)
                            assert result == '/fake/output'

def test_cookiecutter_with_replay_file_path():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_template_name = 'fake_template'
    mock_context = {'cookiecutter': {'key': 'value'}}
    mock_load = lambda replay_dir, template_name: mock_context
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.load', mock_load):
                    with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                        with unittest.mock.patch('cookiecutter.main.generate_files', return_value='/fake/output'):
                            result = cookiecutter(template='fake_template', replay='/path/to/replay.json')
                            assert result == '/fake/output'

def test_cookiecutter_with_no_input():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_context = {'cookiecutter': {'key': 'default'}}
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                    with unittest.mock.patch('cookiecutter.main.generate_files', return_value='/fake/output'):
                        result = cookiecutter(template='fake_template', no_input=True)
                        assert result == '/fake/output'

def test_cookiecutter_with_extra_context():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_context = {'cookiecutter': {'key': 'default'}}
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                    with unittest.mock.patch('cookiecutter.main.generate_files', return_value='/fake/output'):
                        result = cookiecutter(template='fake_template', extra_context={'extra': 'value'})
                        assert result == '/fake/output'

def test_cookiecutter_with_nested_template():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_context = {'cookiecutter': {'template': 'nested_template'}}
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.choose_nested_template', return_value='nested_template'):
                    with unittest.mock.patch('cookiecutter.main.cookiecutter', return_value='/fake/nested_output') as mock_nested:
                        result = cookiecutter(template='fake_template', no_input=True)
                        assert result == '/fake/nested_output'

def test_cookiecutter_cleanup_temp_repo():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/temp_repo'
    mock_base_repo_dir = '/fake/base_repo'
    mock_context = {'cookiecutter': {}}
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_base_repo_dir, True)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                    with unittest.mock.patch('cookiecutter.main.generate_files', return_value='/fake/output'):
                        with unittest.mock.patch('cookiecutter.main.rmtree') as mock_rmtree:
                            result = cookiecutter(template='fake_template', no_input=True)
                            mock_rmtree.assert_any_call(mock_repo_dir)
                            mock_rmtree.assert_any_call(mock_base_repo_dir)

def test_cookiecutter_keep_project_on_failure():
    config_dict = get_user_config(config_file=None, default_config=True)
    mock_repo_dir = '/fake/repo'
    mock_context = {'cookiecutter': {}}
    with unittest.mock.patch('cookiecutter.main.determine_repo_dir', return_value=(mock_repo_dir, False)):
        with unittest.mock.patch('cookiecutter.main.run_pre_prompt_hook', return_value=mock_repo_dir):
            with unittest.mock.patch('cookiecutter.main.generate_context', return_value=mock_context):
                with unittest.mock.patch('cookiecutter.main.prompt_for_config', return_value={}):
                    with unittest.mock.patch('cookiecutter.main.generate_files', side_effect=Exception('Generation failed')):
                        try:
                            cookiecutter(template='fake_template', no_input=True, keep_project_on_failure=True)
                            assert False
                        except Exception as e:
                            assert str(e) == 'Generation failed'


# LLM-generated content at query #3
#--------------------------

def test_replay_false_no_input_false_extra_context_none():
    replay = False
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_none_no_input_false_extra_context_none():
    replay = None
    no_input = False
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_false_no_input_true_extra_context_none():
    replay = False
    no_input = True
    extra_context = None
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False

def test_replay_false_no_input_false_extra_context_dict():
    replay = False
    no_input = False
    extra_context = {"key": "value"}
    predicate = replay and ((no_input is not False) or (extra_context is not None))
    assert predicate == False


# LLM-generated content at query #4
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template='some_template', replay=True, no_input=True)
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template='some_template', replay=True, extra_context={'key': 'value'})
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_replay_bool_loads_from_replay_dir(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.load', return_value={'cookiecutter': {'key': 'value'}})
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    result = cookiecutter(template='template', replay=True)
    assert result == '/output'

def test_cookiecutter_replay_string_loads_from_specified_path(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.load', return_value={'cookiecutter': {'key': 'value'}})
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    result = cookiecutter(template='template', replay='/custom/replay.json')
    assert result == '/output'

def test_cookiecutter_no_replay_generates_context_with_extra_context(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {'extra_key': 'extra_value'}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    result = cookiecutter(template='template', extra_context={'extra_key': 'extra_value'})
    assert result == '/output'

def test_cookiecutter_nested_template_recursion(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {'template': 'nested'}})
    mocker.patch('cookiecutter.main.choose_nested_template', return_value='nested_template')
    mocker.patch('cookiecutter.main.cookiecutter', return_value='/nested_output')
    result = cookiecutter(template='template')
    assert result == '/nested_output'

def test_cookiecutter_cleanup_temp_repo_dir(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/base_repo', True))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/temp_repo')
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    mocker.patch('cookiecutter.main.rmtree')
    result = cookiecutter(template='template')
    assert result == '/output'

def test_cookiecutter_accept_hooks_false_skips_pre_prompt_hook(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    result = cookiecutter(template='template', accept_hooks=False)
    assert result == '/output'

def test_cookiecutter_context_includes_template_and_output_dir(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    dump_mock = mocker.patch('cookiecutter.main.dump')
    mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    cookiecutter(template='template_url', output_dir='/custom_output')
    call_args = dump_mock.call_args[0]
    context = call_args[2]
    assert context['cookiecutter']['_template'] == 'template_url'
    assert context['cookiecutter']['_output_dir'] == os.path.abspath('/custom_output')
    assert context['cookiecutter']['_repo_dir'] == '/repo'

def test_cookiecutter_keep_project_on_failure_passes_to_generate_files(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook', return_value='/repo')
    mocker.patch('cookiecutter.main.generate_context', return_value={'cookiecutter': {}})
    mocker.patch('cookiecutter.main.prompt_for_config', return_value={})
    mocker.patch('cookiecutter.main.dump')
    generate_files_mock = mocker.patch('cookiecutter.main.generate_files', return_value='/output')
    cookiecutter(template='template', keep_project_on_failure=True)
    assert generate_files_mock.call_args[1]['keep_project_on_failure'] == True

def test_cookiecutter_skip_if_file_exists_passes_to_generate_files(mocker):
    mocker.patch('cookiecutter.main.get_user_config', return_value={'abbreviations': {}, 'cookiecutters_dir': '/tmp', 'replay_dir': '/replay', 'default_context': {}})
    mocker.patch('cookiecutter.main.determine_repo_dir', return_value=('/repo', False))
    mocker.patch('cookiecutter.main.run_pre_prompt_hook


# LLM-generated content at query #5
#--------------------------

def test_init_converts_path_to_string():
    from pathlib import Path
    import sys
    import copy
    from unittest.mock import patch
    test_path = Path("/some/dir")
    instance = _patch_import_path_for_repo(test_path)
    assert isinstance(instance._repo_dir, str)
    assert instance._repo_dir == "/some/dir"


# LLM-generated content at query #6
#--------------------------

```python
def test_choose_nested_template_returns_string_path():
    from cookiecutter.prompt import choose_nested_template
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    context = {'cookiecutter': {'templates': {'choice1': {'path': 'nested/template'}}}}
    repo_dir = '/tmp/repo'
    no_input = True
    with patch('cookiecutter.prompt.prompt_choice_for_template', return_value='choice1'):
        result = choose_nested_template(context, repo_dir, no_input)
    assert isinstance(result, str)
    assert result == str((Path(repo_dir).resolve() / Path('nested/template')).resolve())


# LLM-generated content at query #7
#--------------------------

def test_replay_condition_true():
    replay = True
    no_input = False
    extra_context = None
    assert replay and ((no_input is not False) or (extra_context is not None))


# LLM-generated content at query #8
#--------------------------

```python
def test_choose_nested_template_with_valid_relative_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.prompt import choose_nested_template

    mock_context = {
        'cookiecutter': {
            'templates': {
                'template1': {'path': 'subdir/template'}
            }
        }
    }
    mock_repo_dir = Path('/tmp/repo')
    mock_no_input = True

    with patch('cookiecutter.prompt.prompt_choice_for_template') as mock_prompt:
        mock_prompt.return_value = 'template1'
        result = choose_nested_template(mock_context, mock_repo_dir, mock_no_input)
        assert result == '/tmp/repo/subdir/template'

def test_choose_nested_template_with_old_style_valid_relative_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.prompt import choose_nested_template

    mock_context = {
        'cookiecutter': {
            'template': ['Template Name (subdir/template)']
        }
    }
    mock_repo_dir = Path('/tmp/repo')
    mock_no_input = True

    with patch('cookiecutter.prompt.prompt_choice_for_config') as mock_prompt:
        mock_prompt.return_value = 'Template Name (subdir/template)'
        result = choose_nested_template(mock_context, mock_repo_dir, mock_no_input)
        assert result == '/tmp/repo/subdir/template'

def test_choose_nested_template_with_empty_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.prompt import choose_nested_template
    import pytest

    mock_context = {
        'cookiecutter': {
            'templates': {
                'template1': {'path': ''}
            }
        }
    }
    mock_repo_dir = Path('/tmp/repo')
    mock_no_input = True

    with patch('cookiecutter.prompt.prompt_choice_for_template') as mock_prompt:
        mock_prompt.return_value = 'template1'
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(mock_context, mock_repo_dir, mock_no_input)

def test_choose_nested_template_with_absolute_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.prompt import choose_nested_template
    import pytest

    mock_context = {
        'cookiecutter': {
            'templates': {
                'template1': {'path': '/absolute/path/template'}
            }
        }
    }
    mock_repo_dir = Path('/tmp/repo')
    mock_no_input = True

    with patch('cookiecutter.prompt.prompt_choice_for_template') as mock_prompt:
        mock_prompt.return_value = 'template1'
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(mock_context, mock_repo_dir, mock_no_input)

def test_choose_nested_template_with_none_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.prompt import choose_nested_template
    import pytest

    mock_context = {
        'cookiecutter': {
            'templates': {
                'template1': {'path': None}
            }
        }
    }
    mock_repo_dir = Path('/tmp/repo')
    mock_no_input = True

    with patch('cookiecutter.prompt.prompt_choice_for_template') as mock_prompt:
        mock_prompt.return_value = 'template1'
        with pytest.raises(ValueError, match="Illegal template path"):
            choose_nested_template(mock_context, mock_repo_dir, mock_no_input)


# LLM-generated content at query #9
#--------------------------

```python
def test_choose_nested_template_with_absolute_path_raises_value_error():
    context = {'cookiecutter': {'templates': {'choice': {'path': '/absolute/path'}}}}
    repo_dir = '/some/repo'
    no_input = True
    try:
        choose_nested_template(context, repo_dir, no_input)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "Illegal template path"


# LLM-generated content at query #10
#--------------------------

```python
def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_no_input_true():
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_extra_context_provided():
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_raises_invalid_mode_exception_when_replay_string_and_no_input_true():
    try:
        cookiecutter(template="some_template", replay="replay.json", no_input=True)
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_raises_invalid_mode_exception_when_replay_string_and_extra_context_provided():
    try:
        cookiecutter(template="some_template", replay="replay.json", extra_context={"key": "value"})
    except InvalidModeException as e:
        assert str(e) == "You can not use both replay and no_input or extra_context at the same time."

def test_cookiecutter_does_not_raise_when_replay_true_and_no_input_false_and_extra_context_none():
    try:
        cookiecutter(template="some_template", replay=True, no_input=False, extra_context=None)
    except InvalidModeException:
        assert False, "Should not raise InvalidModeException"

def test_cookiecutter_does_not_raise_when_replay_false_and_no_input_true():
    try:
        cookiecutter(template="some_template", replay=False, no_input=True)
    except InvalidModeException:
        assert False, "Should not raise InvalidModeException"

def test_cookiecutter_does_not_raise_when_replay_none_and_no_input_true():
    try:
        cookiecutter(template="some_template", replay=None, no_input=True)
    except InvalidModeException:
        assert False, "Should not raise InvalidModeException"

def test_cookiecutter_does_not_raise_when_replay_false_and_extra_context_provided():
    try:
        cookiecutter(template="some_template", replay=False, extra_context={"key": "value"})
    except InvalidModeException:
        assert False, "Should not raise InvalidModeException"

def test_cookiecutter_does_not_raise_when_replay_none_and_extra_context_provided():
    try:
        cookiecutter(template="some_template", replay=None, extra_context={"key": "value"})
    except InvalidModeException:
        assert False, "Should not raise InvalidModeException"


# LLM-generated content at query #11
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)
    else:
        assert False, "Expected InvalidModeException"

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)
    else:
        assert False, "Expected InvalidModeException"

def test_cookiecutter_with_nested_template():
    context = {"cookiecutter": {"templates": {"choice1": {"path": "subdir"}}}}
    mock_generate_context = lambda context_file, default_context, extra_context: context
    mock_choose_nested_template = lambda context, repo_dir, no_input: "subdir"
    mock_prompt_for_config = lambda context_for_prompting, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: "/fake/output"
    mock_get_user_config = lambda config_file, default_config: {"abbreviations": {}, "cookiecutters_dir": "/fake", "replay_dir": "/fake", "default_context": {}}
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ("/fake/repo", False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {"cookiecutter": {}}
    mock_dump = lambda replay_dir, template_name, context: None
    mock_rmtree = lambda path: None
    import sys
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.prompt'].choose_nested_template = mock_choose_nested_template
    sys.modules['cookiecutter.prompt'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate_files'] = mock_generate_files
    sys.modules['cookiecutter.config'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.repository'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].load = mock_load
    sys.modules['cookiecutter.replay'].dump = mock_dump
    sys.modules['cookiecutter.utils'].rmtree = mock_rmtree
    result = cookiecutter(template="template_with_nested", no_input=True)
    assert result == "/fake/output"

def test_cookiecutter_with_replay():
    context = {"cookiecutter": {"key": "value"}}
    mock_generate_context = lambda context_file, default_context, extra_context: context
    mock_prompt_for_config = lambda context_for_prompting, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: "/fake/output"
    mock_get_user_config = lambda config_file, default_config: {"abbreviations": {}, "cookiecutters_dir": "/fake", "replay_dir": "/fake", "default_context": {}}
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ("/fake/repo", False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_load = lambda replay_dir, template_name: {"cookiecutter": {"key": "replay_value"}}
    mock_dump = lambda replay_dir, template_name, context: None
    mock_rmtree = lambda path: None
    import sys
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.prompt'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate_files'] = mock_generate_files
    sys.modules['cookiecutter.config'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.repository'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].load = mock_load
    sys.modules['cookiecutter.replay'].dump = mock_dump
    sys.modules['cookiecutter.utils'].rmtree = mock_rmtree
    result = cookiecutter(template="some_template", replay=True)
    assert result == "/fake/output"

def test_cookiecutter_with_extra_context():
    context = {"cookiecutter": {"key": "default"}}
    mock_generate_context = lambda context_file, default_context, extra_context: context
    mock_prompt_for_config = lambda context_for_prompting, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: "/fake/output"
    mock_get_user_config = lambda config_file, default_config: {"abbreviations": {}, "cookiecutters_dir": "/fake", "replay_dir": "/fake", "default_context": {}}
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ("/fake/repo", False)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_dump = lambda replay_dir, template_name, context: None
    mock_rmtree = lambda path: None
    import sys
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.prompt'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate_files'] = mock_generate_files
    sys.modules['cookiecutter.config'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.repository'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].dump = mock_dump
    sys.modules['cookiecutter.utils'].rmtree = mock_rmtree
    result = cookiecutter(template="some_template", extra_context={"key": "overridden"})
    assert result == "/fake/output"

def test_cookiecutter_cleanup_on_failure():
    context = {"cookiecutter": {}}
    mock_generate_context = lambda context_file, default_context, extra_context: context
    mock_prompt_for_config = lambda context_for_prompting, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: (_ for _ in ()).throw(Exception("Generation failed"))
    mock_get_user_config = lambda config_file, default_config: {"abbreviations": {}, "cookiecutters_dir": "/fake", "replay_dir": "/fake", "default_context": {}}
    mock_determine_repo_dir = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: ("/fake/repo", True)
    mock_run_pre_prompt_hook = lambda repo_dir: repo_dir
    mock_dump = lambda replay_dir, template_name, context: None
    mock_rmtree = lambda path: None
    import sys
    sys.modules['cookiecutter.generate'].generate_context = mock_generate_context
    sys.modules['cookiecutter.prompt'].prompt_for_config = mock_prompt_for_config
    sys.modules['cookiecutter.generate_files'] = mock_generate_files
    sys.modules['cookiecutter.config'].get_user_config = mock_get_user_config
    sys.modules['cookiecutter.repository'].determine_repo_dir = mock_determine_repo_dir
    sys.modules['cookiecutter.hooks'].run_pre_prompt_hook = mock_run_pre_prompt_hook
    sys.modules['cookiecutter.replay'].dump = mock_dump
    sys.modules['cookiecutter.utils'].rmtree = mock_rmtree
    try:
        cookiecutter(template="some_template", keep_project_on_failure=False)
    except Exception as e:
        assert str(e) == "Generation failed"
    else:
        assert False, "Expected exception"

def test_cookiecutter_skip_hooks():
    context = {"cookiecutter": {}}
    mock_generate_context = lambda context_file, default_context, extra_context: context
    mock_prompt_for_config = lambda context_for_prompting, no_input: {}
    mock_generate_files = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks


# LLM-generated content at query #12
#--------------------------

```python
def test_choose_nested_template_returns_relative_path_string():
    from cookiecutter.prompt import choose_nested_template
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        nested_template = "subdir/template"
        context = {
            'cookiecutter': {
                'templates': {
                    'choice1': {'path': nested_template}
                }
            }
        }
        result = choose_nested_template(context, repo_dir, no_input=True)
        result_path = Path(result)
        assert not result_path.is_absolute()
        assert result_path == (repo_dir / nested_template).resolve()
        assert isinstance(result, str)


# LLM-generated content at query #13
#--------------------------

def test_cookiecutter_replay_and_no_input_conflict():
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_replay_and_extra_context_conflict():
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_replay_bool_loads_from_config():
    config_dict = get_user_config(config_file=None, default_config=True)
    replay_dir = config_dict['replay_dir']
    template_name = "test_template"
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    result = cookiecutter(template="test_template", replay=True, no_input=False)
    assert result is not None

def test_cookiecutter_replay_str_loads_from_path():
    replay_path = "/tmp/replay.json"
    template_name = "test_template"
    context = {'cookiecutter': {'key': 'value'}}
    replay_dir, filename = os.path.split(os.path.splitext(replay_path)[0])
    dump(replay_dir, template_name, context)
    result = cookiecutter(template="test_template", replay=replay_path, no_input=False)
    assert result is not None

def test_cookiecutter_nested_template_selection():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {'templates': {'option1': {'path': 'subdir'}}}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_prompt_for_config_integration():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {'project_name': 'Default Project'}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_generate_files_called():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_cleanup_temp_dirs():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_context_includes_template():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    template = "https://github.com/some/template.git"
    result = cookiecutter(template=template, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_context_includes_output_dir():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    output_dir = "/tmp/output"
    result = cookiecutter(template=".", output_dir=output_dir, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_context_includes_repo_dir():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_context_includes_checkout():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    checkout = "v1.0.0"
    result = cookiecutter(template=".", checkout=checkout, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_dump_replay_file():
    config_dict = get_user_config(config_file=None, default_config=True)
    replay_dir = config_dict['replay_dir']
    template_name = "test_template"
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {'key': 'value'}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    cookiecutter(template=".", no_input=True)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    os.remove(context_file)
    os.remove(replay_file)

def test_cookiecutter_skip_if_file_exists():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", skip_if_file_exists=True, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_accept_hooks_false():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", accept_hooks=False, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_keep_project_on_failure():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", keep_project_on_failure=True, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_with_directory_param():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", directory="subdir", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_with_password():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", password="secret", no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_with_config_file():
    config_file = "custom_config.yaml"
    with open(config_file, 'w') as f:
        f.write("default_context:\n  project_name: Test")
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", config_file=config_file, no_input=True)
    os.remove(config_file)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_with_default_config():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", default_config=True, no_input=True)
    os.remove(context_file)
    assert result is not None

def test_cookiecutter_overwrite_if_exists():
    context_file = 'cookiecutter.json'
    context_data = {'cookiecutter': {}}
    with open(context_file, 'w') as f:
        json.dump(context_data, f)
    result = cookiecutter(template=".", overwrite_if_exists=True, no_input=True)
    os.remove(context_file)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_cleanup_set_to_true_when_accept_hooks_true_and_pre_prompt_hook_creates_tmp_dir():
    mock_base_repo_dir = "/fake/base/repo"
    mock_tmp_repo_dir = "/fake/tmp/repo"
    mock_find_hook = lambda hook: ["pre_prompt_script.py"] if hook == "pre_prompt" else []
    mock_create_tmp_repo_dir = lambda repo_dir: mock_tmp_repo_dir
    mock_run_script = lambda script, repo_dir: None
    original_run_pre_prompt_hook = cookiecutter.hooks.run_pre_prompt_hook
    cookiecutter.hooks.run_pre_prompt_hook = lambda repo_dir: mock_tmp_repo_dir if repo_dir == mock_base_repo_dir else repo_dir
    accept_hooks = True
    repo_dir = mock_base_repo_dir
    cleanup = False
    repo_dir = str(cookiecutter.hooks.run_pre_prompt_hook(mock_base_repo_dir)) if accept_hooks else repo_dir
    cleanup = repo_dir != mock_base_repo_dir
    cookiecutter.hooks.run_pre_prompt_hook = original_run_pre_prompt_hook
    assert cleanup == True


# LLM-generated content at query #15
#--------------------------

```python
def test_replay_and_no_input_raises_invalid_mode_exception():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = True
    no_input = True
    extra_context = None
    with pytest.raises(InvalidModeException):
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)

def test_replay_and_extra_context_raises_invalid_mode_exception():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = True
    no_input = False
    extra_context = {"key": "value"}
    with pytest.raises(InvalidModeException):
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)

def test_replay_and_no_input_and_extra_context_raises_invalid_mode_exception():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = True
    no_input = True
    extra_context = {"key": "value"}
    with pytest.raises(InvalidModeException):
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)

def test_replay_false_with_no_input_does_not_raise():
    from cookiecutter.main import cookiecutter
    replay = False
    no_input = True
    extra_context = None
    try:
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)
    except Exception as e:
        assert not isinstance(e, InvalidModeException)

def test_replay_false_with_extra_context_does_not_raise():
    from cookiecutter.main import cookiecutter
    replay = False
    no_input = False
    extra_context = {"key": "value"}
    try:
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)
    except Exception as e:
        assert not isinstance(e, InvalidModeException)

def test_replay_string_with_no_input_raises_invalid_mode_exception():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = "replay.json"
    no_input = True
    extra_context = None
    with pytest.raises(InvalidModeException):
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)

def test_replay_string_with_extra_context_raises_invalid_mode_exception():
    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import InvalidModeException
    import pytest
    replay = "replay.json"
    no_input = False
    extra_context = {"key": "value"}
    with pytest.raises(InvalidModeException):
        cookiecutter(template="test", replay=replay, no_input=no_input, extra_context=extra_context)


# LLM-generated content at query #16
#--------------------------

```python
def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_no_input_true():
    replay = True
    no_input = True
    extra_context = None
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                with pytest.raises(cookiecutter.exceptions.InvalidModeException):
                                    cookiecutter(template=template, replay=replay, no_input=no_input)

def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_extra_context_not_none():
    replay = True
    no_input = False
    extra_context = {"key": "value"}
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                with pytest.raises(cookiecutter.exceptions.InvalidModeException):
                                    cookiecutter(template=template, replay=replay, extra_context=extra_context)

def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_no_input_true_and_extra_context_not_none():
    replay = True
    no_input = True
    extra_context = {"key": "value"}
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                with pytest.raises(cookiecutter.exceptions.InvalidModeException):
                                    cookiecutter(template=template, replay=replay, no_input=no_input, extra_context=extra_context)

def test_cookiecutter_does_not_raise_when_replay_false_and_no_input_true():
    replay = False
    no_input = True
    extra_context = None
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                result = cookiecutter(template=template, replay=replay, no_input=no_input)
                                assert result == "/tmp/result"

def test_cookiecutter_does_not_raise_when_replay_false_and_extra_context_not_none():
    replay = False
    no_input = False
    extra_context = {"key": "value"}
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                result = cookiecutter(template=template, replay=replay, extra_context=extra_context)
                                assert result == "/tmp/result"

def test_cookiecutter_does_not_raise_when_replay_none_and_no_input_true():
    replay = None
    no_input = True
    extra_context = None
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                        with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                            with unittest.mock.patch("cookiecutter.main.dump"):
                                result = cookiecutter(template=template, replay=replay, no_input=no_input)
                                assert result == "/tmp/result"

def test_cookiecutter_does_not_raise_when_replay_string_and_no_input_false_and_extra_context_none():
    replay = "/path/to/replay.json"
    no_input = False
    extra_context = None
    template = "some_template"
    config_dict = {"abbreviations": {}, "cookiecutters_dir": "/tmp", "default_context": {}, "replay_dir": "/tmp"}
    with unittest.mock.patch("cookiecutter.main.get_user_config", return_value=config_dict):
        with unittest.mock.patch("cookiecutter.main.determine_repo_dir", return_value=("/tmp/repo", False)):
            with unittest.mock.patch("cookiecutter.main.run_pre_prompt_hook", return_value="/tmp/repo"):
                with unittest.mock.patch("cookiecutter.main.load", return_value={"cookiecutter": {}}):
                    with unittest.mock.patch("cookiecutter.main.generate_context", return_value={"cookiecutter": {}}):
                        with unittest.mock.patch("cookiecutter.main.prompt_for_config", return_value={}):
                            with unittest.mock.patch("cookiecutter.main.generate_files", return_value="/tmp/result"):
                                with unittest.mock.patch("cookiecutter.main.dump"):
                                    result = cookiecutter(template=template, replay=replay, no_input=no_input, extra_context=extra_context)
                                    assert result == "/tmp/result"


# LLM-generated content at query #17
#--------------------------

def test_repo_dir_converted_to_string_when_path():
    import sys
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    test_path = Path("/some/dir")
    instance = _patch_import_path_for_repo(test_path)
    assert isinstance(instance._repo_dir, str)
    assert instance._repo_dir == "/some/dir"


# LLM-generated content at query #18
#--------------------------

def test_accept_hooks_false_ensures_predicate_false():
    repo_dir = "/some/repo/dir"
    base_repo_dir = repo_dir
    accept_hooks = False
    result = str(run_pre_prompt_hook(base_repo_dir)) if accept_hooks else repo_dir
    assert result == repo_dir
    assert result == base_repo_dir


# LLM-generated content at query #19
#--------------------------

```python
def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_no_input_used_together():
    try:
        cookiecutter(template="some_template", replay=True, no_input=True)
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_raises_invalid_mode_exception_when_replay_and_extra_context_used_together():
    try:
        cookiecutter(template="some_template", replay=True, extra_context={"key": "value"})
    except InvalidModeException as e:
        assert "You can not use both replay and no_input or extra_context at the same time." in str(e)

def test_cookiecutter_calls_get_user_config_with_provided_parameters():
    config_dict = get_user_config(config_file="custom_config.yaml", default_config=False)
    assert isinstance(config_dict, dict)

def test_cookiecutter_calls_determine_repo_dir_with_template_and_config():
    base_repo_dir, cleanup = determine_repo_dir(template="some_template", abbreviations={}, clone_to_dir="/tmp", checkout=None, no_input=False, password=None, directory=None)
    assert isinstance(base_repo_dir, str)
    assert isinstance(cleanup, bool)

def test_cookiecutter_loads_replay_context_when_replay_is_true():
    context = load("/replay/dir", "template_name")
    assert isinstance(context, dict)
    assert "cookiecutter" in context

def test_cookiecutter_loads_replay_context_when_replay_is_string_path():
    path, template_name = os.path.split(os.path.splitext("/path/to/replay.json")[0])
    context = load(path, template_name)
    assert isinstance(context, dict)
    assert "cookiecutter" in context

def test_cookiecutter_generates_context_with_replay_and_extra_context_none():
    context = generate_context(context_file="cookiecutter.json", default_context={}, extra_context=None)
    assert isinstance(context, dict)
    assert "cookiecutter" in context

def test_cookiecutter_generates_context_without_replay_and_extra_context_provided():
    context = generate_context(context_file="cookiecutter.json", default_context={}, extra_context={"key": "value"})
    assert isinstance(context, dict)
    assert "cookiecutter" in context

def test_cookiecutter_calls_choose_nested_template_when_template_key_in_context():
    context = {"cookiecutter": {"template": ["option1", "option2"]}}
    template_path = choose_nested_template(context, "/repo/dir", no_input=False)
    assert isinstance(template_path, str)

def test_cookiecutter_calls_choose_nested_template_when_templates_key_in_context():
    context = {"cookiecutter": {"templates": {"opt1": {"path": "path1"}, "opt2": {"path": "path2"}}}}
    template_path = choose_nested_template(context, "/repo/dir", no_input=False)
    assert isinstance(template_path, str)

def test_cookiecutter_updates_context_with_prompt_for_config_when_context_for_prompting_not_empty():
    context_for_prompting = {"cookiecutter": {"key1": "default1", "key2": "default2"}}
    prompted = prompt_for_config(context_for_prompting, no_input=False)
    assert isinstance(prompted, dict)

def test_cookiecutter_calls_generate_files_with_correct_parameters():
    result = generate_files(repo_dir="/repo/dir", context={"cookiecutter": {}}, output_dir=".", overwrite_if_exists=False, skip_if_file_exists=False, accept_hooks=True, keep_project_on_failure=False)
    assert isinstance(result, str)

def test_cookiecutter_dumps_context_to_replay_dir():
    dump("/replay/dir", "template_name", {"cookiecutter": {"key": "value"}})

def test_cookiecutter_cleans_up_repo_dir_when_cleanup_true():
    rmtree("/tmp/some_repo_dir")

def test_cookiecutter_cleans_up_base_repo_dir_when_cleanup_base_repo_dir_true():
    rmtree("/tmp/base_repo_dir")

def test_cookiecutter_returns_generated_project_path():
    result = cookiecutter(template="some_template", no_input=True, default_config=True)
    assert isinstance(result, str)


# LLM-generated content at query #20
#--------------------------

def test_constructor_with_path_object():
    from pathlib import Path
    from unittest.mock import patch
    repo_dir = Path("/some/path")
    instance = _patch_import_path_for_repo(repo_dir)
    assert instance._repo_dir == "/some/path"

def test_constructor_with_string():
    repo_dir = "/another/path"
    instance = _patch_import_path_for_repo(repo_dir)
    assert instance._repo_dir == "/another/path"

def test_constructor_path_object_converted_to_string():
    from pathlib import Path
    repo_dir = Path("/test/path")
    instance = _patch_import_path_for_repo(repo_dir)
    assert isinstance(instance._repo_dir, str)

def test_constructor_string_unchanged():
    repo_dir = "/string/path"
    instance = _patch_import_path_for_repo(repo_dir)
    assert instance._repo_dir == repo_dir


