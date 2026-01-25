####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = "valid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("replay_dir: $HOME/replays\ncookiecutters_dir: ~/cookiecutters")
    config = get_config(config_path)
    os.remove(config_path)
    assert isinstance(config, dict)
    assert "replay_dir" in config
    assert "cookiecutters_dir" in config
    assert config["replay_dir"] == os.path.expanduser(os.path.expandvars("$HOME/replays"))
    assert config["cookiecutters_dir"] == os.path.expanduser("~/cookiecutters")

def test_get_config_with_non_existent_file():
    config_path = "non_existent_config.yaml"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml_file():
    config_path = "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'
    os.remove(config_path)

def test_get_config_with_non_dict_top_level_yaml_file():
    config_path = "non_dict_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("- item1\n- item2")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'
    os.remove(config_path)

def test_get_config_merges_with_default_config():
    config_path = "merge_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("replay_dir: $HOME/replays\ncookiecutters_dir: ~/cookiecutters")
    config = get_config(config_path)
    os.remove(config_path)
    assert all(key in config for key in DEFAULT_CONFIG.keys())
    assert config["replay_dir"] == os.path.expanduser(os.path.expandvars("$HOME/replays"))
    assert config["cookiecutters_dir"] == os.path.expanduser("~/cookiecutters")


# LLM-generated content at query #2
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookies'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'custom_config.yml'
    config_file.write_text('replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies')
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/custom/replay'
    assert result['cookiecutters_dir'] == '/custom/cookies'


def test_get_user_config_with_env_var_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yml'
    config_file.write_text('replay_dir: /env/replay\ncookiecutters_dir: /env/cookies')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/replay'
    assert result['cookiecutters_dir'] == '/env/cookies'


def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    user_config = tmp_path / '.cookiecutterrc'
    user_config.write_text('replay_dir: /user/replay\ncookiecutters_dir: /user/cookies')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config))
    result = get_user_config()
    assert result['replay_dir'] == '/user/replay'
    assert result['cookiecutters_dir'] == '/user/cookies'


def test_get_user_config_fallback_to_default():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #3
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    get_user_config()
    assert 'KeyError' not in [type(e).__name__ for e in pytest.raises(Exception)]


# LLM-generated content at query #4
#--------------------------

```python
def test_config_path_exists():
    config_path = "existing_config.yaml"
    os.path.exists = lambda x: True
    assert get_config(config_path) is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_config_file_exists():
    config_path = "existing_config.yaml"
    os.path.exists = lambda path: True
    yaml.safe_load = lambda file_handle: {}
    result = get_config(config_path)
    assert isinstance(result, dict)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    USER_CONFIG_PATH = "/path/to/existing/config"
    assert os.path.exists(USER_CONFIG_PATH) == True


# LLM-generated content at query #7
#--------------------------

```
def test_get_config_raises_exception_when_config_path_does_not_exist():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    non_existent_path = Path('/non/existent/path')
    with patch('os.path.exists', return_value=False):
        try:
            get_config(non_existent_path)
            assert False, "Expected ConfigDoesNotExistException to be raised"
        except ConfigDoesNotExistException as e:
            assert str(e) == f'Config file {non_existent_path} does not exist.'


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #9
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookies'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'config.yml'
    config_file.write_text('replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies')
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/custom/replay'
    assert result['cookiecutters_dir'] == '/custom/cookies'


def test_get_user_config_with_env_var_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yml'
    config_file.write_text('replay_dir: /env/replay\ncookiecutters_dir: /env/cookies')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/replay'
    assert result['cookiecutters_dir'] == '/env/cookies'


def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    user_config = tmp_path / '.cookiecutter.yaml'
    user_config.write_text('replay_dir: /user/replay\ncookiecutters_dir: /user/cookies')
    monkeypatch.setattr('os.path.expanduser', lambda x: str(user_config) if x == '~/.cookiecutter.yaml' else x)
    result = get_user_config()
    assert result['replay_dir'] == '/user/replay'
    assert result['cookiecutters_dir'] == '/user/cookies'


def test_get_user_config_fallback_to_default():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #10
#--------------------------

```python
def test_expand_path_with_env_var():
    os.environ["TEST_VAR"] = "/test"
    assert _expand_path("$TEST_VAR/path") == "/test/path"

def test_expand_path_with_user_home():
    assert _expand_path("~/path") == os.path.expanduser("~/path")

def test_expand_path_with_both_env_var_and_user_home():
    os.environ["TEST_VAR"] = "/test"
    assert _expand_path("$TEST_VAR/~/path") == "/test/~/path"

def test_expand_path_without_env_var_or_user_home():
    assert _expand_path("/some/path") == "/some/path"


# LLM-generated content at query #11
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('replay_dir: /custom/path')
        tmp.flush()
        result = get_user_config(config_file=tmp.name)
        assert result['replay_dir'] == '/custom/path'

def test_get_user_config_env_var_config():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('replay_dir: /env/path')
        tmp.flush()
        os.environ['COOKIECUTTER_CONFIG'] = tmp.name
        result = get_user_config()
        assert result['replay_dir'] == '/env/path'
        del os.environ['COOKIECUTTER_CONFIG']

def test_get_user_config_user_config_exists():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('replay_dir: /user/path')
        tmp.flush()
        original_path = USER_CONFIG_PATH
        USER_CONFIG_PATH = tmp.name
        result = get_user_config()
        assert result['replay_dir'] == '/user/path'
        USER_CONFIG_PATH = original_path

def test_get_user_config_fallback_to_default():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false_when_config_file_is_none():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)

def test_predicate_at_line_33_evaluates_to_false_when_config_file_is_user_config_path():
    config_file = USER_CONFIG_PATH
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = USER_CONFIG_PATH
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists = lambda _: True
    result = get_user_config()
    assert isinstance(result, dict)


# LLM-generated content at query #16
#--------------------------

```python
def test_config_file_not_user_config_path():
    config_file = "/path/to/custom/config"
    USER_CONFIG_PATH = "/path/to/user/config"
    assert not (config_file and config_file is not USER_CONFIG_PATH)

def test_config_file_is_user_config_path():
    config_file = "/path/to/user/config"
    USER_CONFIG_PATH = "/path/to/user/config"
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=default_config)
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_custom_config_file(tmpdir):
    config_file = tmpdir.join("config.yml")
    config_file.write("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    result = get_user_config(config_file=str(config_file))
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_env_config_file(tmpdir, monkeypatch):
    config_file = tmpdir.join("config.yml")
    config_file.write("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    monkeypatch.setenv("COOKIECUTTER_CONFIG", str(config_file))
    result = get_user_config()
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config_file(tmpdir):
    config_file = tmpdir.join("config.yml")
    config_file.write("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    monkeypatch.setattr("module.USER_CONFIG_PATH", str(config_file))
    result = get_user_config()
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_without_config_file():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    USER_CONFIG_PATH = "/default/path/to/config"
    config_file = USER_CONFIG_PATH
    result = get_user_config(config_file=config_file)
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #19
#--------------------------

```
def test_predicate_at_line_43_evaluates_to_true():
    USER_CONFIG_PATH = "/path/to/existing/config"
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    assert os.path.exists(USER_CONFIG_PATH) == True


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config_raises_exception_when_config_path_does_not_exist():
    config_path = "non_existent_path"
    try:
        get_config(config_path)
        assert False
    except ConfigDoesNotExistException:
        assert True


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    USER_CONFIG_PATH = "/default/path"
    assert not (config_file and config_file is not USER_CONFIG_PATH)
    
    config_file = "/default/path"
    assert not (config_file and config_file is not USER_CONFIG_PATH)
    
    config_file = "/custom/path"
    USER_CONFIG_PATH = "/default/path"
    assert (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #23
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file(mocker):
    mock_config = {'replay_dir': '~/custom_replay', 'cookiecutters_dir': '~/custom_cookiecutters'}
    mocker.patch('builtins.open', mocker.mock_open(read_data='replay_dir: ~/custom_replay\ncookiecutters_dir: ~/custom_cookiecutters'))
    mocker.patch('yaml.safe_load', return_value=mock_config)
    mocker.patch('os.path.exists', return_value=True)
    result = get_user_config(config_file='/custom/config.yaml')
    assert result['replay_dir'] == os.path.expanduser('~/custom_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')

def test_get_user_config_with_env_var(mocker):
    mock_config = {'replay_dir': '~/env_replay', 'cookiecutters_dir': '~/env_cookiecutters'}
    mocker.patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'})
    mocker.patch('builtins.open', mocker.mock_open(read_data='replay_dir: ~/env_replay\ncookiecutters_dir: ~/env_cookiecutters'))
    mocker.patch('yaml.safe_load', return_value=mock_config)
    mocker.patch('os.path.exists', return_value=True)
    result = get_user_config()
    assert result['replay_dir'] == os.path.expanduser('~/env_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/env_cookiecutters')

def test_get_user_config_with_default_user_config(mocker):
    mock_config = {'replay_dir': '~/default_replay', 'cookiecutters_dir': '~/default_cookiecutters'}
    mocker.patch('builtins.open', mocker.mock_open(read_data='replay_dir: ~/default_replay\ncookiecutters_dir: ~/default_cookiecutters'))
    mocker.patch('yaml.safe_load', return_value=mock_config)
    mocker.patch('os.path.exists', return_value=True)
    result = get_user_config()
    assert result['replay_dir'] == os.path.expanduser('~/default_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/default_cookiecutters')

def test_get_user_config_with_no_config_found(mocker):
    mocker.patch.dict('os.environ', clear=True)
    mocker.patch('os.path.exists', return_value=False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #24
#--------------------------

```python
def test_get_config_valid_yaml():
    test_config_path = "valid_config.yaml"
    with open(test_config_path, "w") as f:
        f.write("replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters")
    config = get_config(test_config_path)
    assert config["replay_dir"] == os.path.expanduser("~/replays")
    assert config["cookiecutters_dir"] == os.path.expanduser("~/cookiecutters")
    os.remove(test_config_path)

def test_get_config_nonexistent_file():
    test_config_path = "nonexistent_config.yaml"
    try:
        get_config(test_config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {test_config_path} does not exist.'

def test_get_config_invalid_yaml():
    test_config_path = "invalid_config.yaml"
    with open(test_config_path, "w") as f:
        f.write("invalid: yaml: [")
    try:
        get_config(test_config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {test_config_path}.'
    os.remove(test_config_path)

def test_get_config_non_dict_yaml():
    test_config_path = "non_dict_config.yaml"
    with open(test_config_path, "w") as f:
        f.write("- item1\n- item2")
    try:
        get_config(test_config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {test_config_path} should be an object.'
    os.remove(test_config_path)

def test_get_config_empty_yaml():
    test_config_path = "empty_config.yaml"
    with open(test_config_path, "w") as f:
        f.write("")
    config = get_config(test_config_path)
    assert config["replay_dir"] == os.path.expanduser(DEFAULT_CONFIG["replay_dir"])
    assert config["cookiecutters_dir"] == os.path.expanduser(DEFAULT_CONFIG["cookiecutters_dir"])
    os.remove(test_config_path)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #26
#--------------------------

```python
def test_get_user_config_default_config_dict():
    default_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=default_config)
    assert result == merge_configs(DEFAULT_CONFIG, default_config)

def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_custom_config_file():
    custom_config_file = '/path/to/custom/config.yaml'
    result = get_user_config(config_file=custom_config_file)
    assert result == get_config(custom_config_file)

def test_get_user_config_env_var_config_file(monkeypatch):
    env_config_file = '/path/to/env/config.yaml'
    monkeypatch.setenv('COOKIECUTTER_CONFIG', env_config_file)
    result = get_user_config()
    assert result == get_config(env_config_file)

def test_get_user_config_user_config_path_exists():
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)

def test_get_user_config_user_config_path_not_exists(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = USER_CONFIG_PATH
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #28
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    result = get_user_config(default_config=custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    config_file = "/custom/config.yaml"
    result = get_user_config(config_file=config_file)
    assert result == get_config(config_file)

def test_get_user_config_env_var():
    os.environ["COOKIECUTTER_CONFIG"] = "/env/config.yaml"
    result = get_user_config()
    assert result == get_config("/env/config.yaml")
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_default_user_config_path():
    result = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        assert result == get_config(USER_CONFIG_PATH)
    else:
        assert result == DEFAULT_CONFIG

def test_get_user_config_user_config_path():
    config_file = USER_CONFIG_PATH
    result = get_user_config(config_file=config_file)
    if os.path.exists(USER_CONFIG_PATH):
        assert result == get_config(USER_CONFIG_PATH)
    else:
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #29
#--------------------------

```python
def test_config_file_exists():
    config_path = "existing_config.yaml"
    assert os.path.exists(config_path)


# LLM-generated content at query #30
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=default_config)
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    result = get_user_config(config_file=str(config_file))
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    monkeypatch.setenv("COOKIECUTTER_CONFIG", str(config_file))
    result = get_user_config()
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    monkeypatch.setattr("os.path.exists", lambda path: path == str(config_file))
    monkeypatch.setattr("os.environ", {})
    result = get_user_config()
    assert result["replay_dir"] == "/custom/replay"
    assert result["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_True():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.environ = {}
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config_with_existing_file():
    test_file = "existing_config.yaml"
    with open(test_file, "w") as f:
        f.write("replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters")
    result = get_config(test_file)
    assert isinstance(result, dict)
    assert result["replay_dir"] == "/path/to/replay"
    assert result["cookiecutters_dir"] == "/path/to/cookiecutters"
    os.remove(test_file)


# LLM-generated content at query #33
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    result = get_user_config()
    assert 'COOKIECUTTER_CONFIG' in os.environ


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    custom_config_file = "/custom/config.yaml"
    result = get_user_config(config_file=custom_config_file)
    assert result == get_config(custom_config_file)

def test_get_user_config_env_config_file():
    os.environ["COOKIECUTTER_CONFIG"] = "/env/config.yaml"
    result = get_user_config()
    assert result == get_config("/env/config.yaml")
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_user_config_path():
    result = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        assert result == get_config(USER_CONFIG_PATH)
    else:
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #2
#--------------------------

```python
def test_expand_path_with_home_dir():
    path = "~"
    expanded = _expand_path(path)
    assert expanded != path
    assert os.path.isabs(expanded)

def test_expand_path_with_env_var():
    os.environ["TEST_PATH"] = "/tmp"
    path = "$TEST_PATH/file.txt"
    expanded = _expand_path(path)
    assert expanded == "/tmp/file.txt"

def test_expand_path_with_both_home_and_env():
    os.environ["TEST_DIR"] = "documents"
    path = "~/$TEST_DIR/file.txt"
    expanded = _expand_path(path)
    assert expanded.startswith(os.path.expanduser("~"))
    assert "documents/file.txt" in expanded

def test_expand_path_with_no_expansion_needed():
    path = "/absolute/path/file.txt"
    expanded = _expand_path(path)
    assert expanded == path

def test_expand_path_with_unknown_env_var():
    path = "$UNKNOWN_VAR/file.txt"
    expanded = _expand_path(path)
    assert expanded == "/file.txt"


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    config = get_user_config()
    assert config == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    USER_CONFIG_PATH = "/path/to/user/config"
    assert not (config_file and config_file is not USER_CONFIG_PATH)

    config_file = "/path/to/user/config"
    assert not (config_file and config_file is not USER_CONFIG_PATH)

    config_file = "/different/path"
    assert (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config_with_valid_yaml():
    test_config_path = 'test_config.yml'
    with open(test_config_path, 'w') as f:
        f.write('replay_dir: ~/test_replay\ncookiecutters_dir: ~/test_cookies\n')
    result = get_config(test_config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result
    os.remove(test_config_path)

def test_get_config_with_invalid_yaml():
    test_config_path = 'test_config.yml'
    with open(test_config_path, 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(test_config_path)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(test_config_path)

def test_get_config_with_nonexistent_file():
    test_config_path = 'nonexistent.yml'
    try:
        get_config(test_config_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_non_dict_yaml():
    test_config_path = 'test_config.yml'
    with open(test_config_path, 'w') as f:
        f.write('- item1\n- item2\n')
    try:
        get_config(test_config_path)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(test_config_path)

def test_get_config_expands_paths():
    test_config_path = 'test_config.yml'
    with open(test_config_path, 'w') as f:
        f.write('replay_dir: ~/test_replay\ncookiecutters_dir: ~/test_cookies\n')
    result = get_config(test_config_path)
    assert not result['replay_dir'].startswith('~')
    assert not result['cookiecutters_dir'].startswith('~')
    os.remove(test_config_path)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists():
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile() as tmp_file:
        USER_CONFIG_PATH = tmp_file.name
        assert os.path.exists(USER_CONFIG_PATH) == True


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_file_exists():
    config_path = "existing_config.yaml"
    os.path.exists = lambda x: True
    yaml.safe_load = lambda x: {"replay_dir": "replays", "cookiecutters_dir": "cookies"}
    DEFAULT_CONFIG = {}
    merge_configs = lambda x, y: y
    _expand_path = lambda x: x
    config_dict = get_config(config_path)
    assert config_dict == {"replay_dir": "replays", "cookiecutters_dir": "cookies"}


# LLM-generated content at query #8
#--------------------------

```
def test_predicate_at_line_43_evaluates_to_true():
    import os
    import tempfile
    from unittest.mock import patch

    # Create a temporary file to simulate USER_CONFIG_PATH
    with tempfile.NamedTemporaryFile() as tmp_file:
        USER_CONFIG_PATH = tmp_file.name
        with patch('os.path.exists', return_value=True):
            assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #9
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    expected = copy.copy(DEFAULT_CONFIG)
    assert result == expected

def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    custom_config_path = '/path/to/custom/config.yaml'
    expected_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    with patch('builtins.open', mock_open(read_data='replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config(config_file=custom_config_path)
        assert result == expected_config

def test_get_user_config_env_var():
    env_config_path = '/path/to/env/config.yaml'
    expected_config = {'replay_dir': '/env/replay', 'cookiecutters_dir': '/env/cookiecutters'}
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': env_config_path}), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/replay\ncookiecutters_dir: /env/cookiecutters')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config()
        assert result == expected_config

def test_get_user_config_default_user_config():
    expected_config = {'replay_dir': '/default/replay', 'cookiecutters_dir': '/default/cookiecutters'}
    with patch('builtins.open', mock_open(read_data='replay_dir: /default/replay\ncookiecutters_dir: /default/cookiecutters')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config()
        assert result == expected_config

def test_get_user_config_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #10
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists():
    import os
    USER_CONFIG_PATH = "/tmp/test_config"
    with open(USER_CONFIG_PATH, "w") as f:
        f.write("test")
    assert os.path.exists(USER_CONFIG_PATH)
    os.remove(USER_CONFIG_PATH)

def test_predicate_at_line_43_evaluates_to_false_when_user_config_path_does_not_exist():
    import os
    USER_CONFIG_PATH = "/tmp/nonexistent_config"
    assert not os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_false():
    result = get_user_config(default_config=False)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    custom_config_file = "/custom/config.yaml"
    mock_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(config_file=custom_config_file)
    expected = merge_configs(DEFAULT_CONFIG, mock_config)
    assert result == expected

def test_get_user_config_with_env_config_file():
    env_config_file = "/env/config.yaml"
    os.environ["COOKIECUTTER_CONFIG"] = env_config_file
    mock_config = {"replay_dir": "/env/replay", "cookiecutters_dir": "/env/cookiecutters"}
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, mock_config)
    assert result == expected
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_with_user_config_path():
    mock_config = {"replay_dir": "/user/replay", "cookiecutters_dir": "/user/cookiecutters"}
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, mock_config)
    assert result == expected

def test_get_user_config_with_invalid_config_file():
    invalid_config_file = "/invalid/config.yaml"
    try:
        get_user_config(config_file=invalid_config_file)
    except ConfigDoesNotExistException:
        assert True
    else:
        assert False

def test_get_user_config_with_invalid_yaml_file():
    invalid_yaml_file = "/invalid/yaml.yaml"
    try:
        get_user_config(config_file=invalid_yaml_file)
    except InvalidConfiguration:
        assert True
    else:
        assert False

def test_get_user_config_with_non_dict_yaml_file():
    non_dict_yaml_file = "/non_dict/yaml.yaml"
    try:
        get_user_config(config_file=non_dict_yaml_file)
    except InvalidConfiguration:
        assert True
    else:
        assert False


# LLM-generated content at query #13
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    try:
        get_user_config()
    except KeyError:
        assert False, "Predicate at line 40 should evaluate to False when COOKIECUTTER_CONFIG is set"
    finally:
        del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #14
#--------------------------

```python
def test_config_file_exists():
    config_path = "existing_config.yml"
    os.path.exists = lambda path: True
    get_config(config_path)


# LLM-generated content at query #15
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=default_config)
    expected = merge_configs(DEFAULT_CONFIG, default_config)
    assert result == expected

def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_custom_config_file():
    config_file = "/path/to/config.yaml"
    result = get_user_config(config_file=config_file)
    expected = get_config(config_file)
    assert result == expected

def test_get_user_config_with_env_config_file():
    os.environ["COOKIECUTTER_CONFIG"] = "/env/config.yaml"
    result = get_user_config()
    expected = get_config("/env/config.yaml")
    assert result == expected
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_with_default_user_config():
    result = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        expected = get_config(USER_CONFIG_PATH)
    else:
        expected = DEFAULT_CONFIG
    assert result == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    config = get_user_config()
    assert config is not None


# LLM-generated content at query #17
#--------------------------

```
def test_get_config_raises_exception_when_config_path_does_not_exist():
    import os
    import pytest
    from unittest.mock import patch
    from pathlib import Path

    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config('nonexistent_path')


# LLM-generated content at query #18
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    config = get_user_config(default_config=default_config)
    assert config["replay_dir"] == "/custom/replay"
    assert config["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config_true():
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    config = get_user_config(config_file=str(config_file))
    assert config["replay_dir"] == "/custom/replay"
    assert config["cookiecutters_dir"] == "/custom/cookiecutters"

def test_get_user_config_with_default_config_file(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters")
    os.environ["COOKIECUTTER_CONFIG"] = str(config_file)
    config = get_user_config()
    assert config["replay_dir"] == "/custom/replay"
    assert config["cookiecutters_dir"] == "/custom/cookiecutters"
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_with_missing_config_file():
    config = get_user_config(config_file="/nonexistent/path/config.yml")
    assert config == DEFAULT_CONFIG

def test_get_user_config_with_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("invalid: yaml: file")
    config = get_user_config(config_file=str(config_file))
    assert config == DEFAULT_CONFIG

def test_get_user_config_with_non_dict_yaml(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("- item1\n- item2")
    config = get_user_config(config_file=str(config_file))
    assert config == DEFAULT_CONFIG


# LLM-generated content at query #19
#--------------------------

```python
def test_get_config_with_valid_file():
    config_file = 'valid_config.yaml'
    config = get_config(config_file)
    assert isinstance(config, dict)
    assert 'replay_dir' in config
    assert 'cookiecutters_dir' in config

def test_get_config_with_nonexistent_file():
    config_file = 'nonexistent_config.yaml'
    try:
        get_config(config_file)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_file} does not exist.'

def test_get_config_with_invalid_yaml():
    config_file = 'invalid_yaml_config.yaml'
    try:
        get_config(config_file)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_file}.'

def test_get_config_with_non_dict_top_level():
    config_file = 'non_dict_top_level_config.yaml'
    try:
        get_config(config_file)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_file} should be an object.'

def test_get_config_with_expanded_paths():
    config_file = 'config_with_paths.yaml'
    config = get_config(config_file)
    assert config['replay_dir'] == os.path.expanduser(os.path.expandvars(config['replay_dir']))
    assert config['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars(config['cookiecutters_dir']))


# LLM-generated content at query #20
#--------------------------

```python
def test_get_user_config_default_config_dict():
    default_config = {"key": "value"}
    result = get_user_config(default_config=default_config)
    assert result == merge_configs(DEFAULT_CONFIG, default_config)

def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_custom_config_file():
    config_file = "/path/to/config"
    result = get_user_config(config_file=config_file)
    assert result == get_config(config_file)

def test_get_user_config_env_config_file():
    env_config_file = "/path/to/env/config"
    os.environ["COOKIECUTTER_CONFIG"] = env_config_file
    result = get_user_config()
    assert result == get_config(env_config_file)

def test_get_user_config_user_config_path_exists():
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)

def test_get_user_config_user_config_path_not_exists():
    if os.path.exists(USER_CONFIG_PATH):
        os.remove(USER_CONFIG_PATH)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #21
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'config.yml'
    config_file.write_text('replay_dir: /custom/replay')
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yml'
    config_file.write_text('replay_dir: /env/replay')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_user_config_exists(tmp_path, monkeypatch):
    user_config = tmp_path / '.cookiecutterrc'
    user_config.write_text('replay_dir: /user/replay')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config))
    result = get_user_config()
    assert result['replay_dir'] == '/user/replay'


def test_get_user_config_fallback_to_default():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #22
#--------------------------

```python
def test_config_file_exists():
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    try:
        os.path.exists(temp_file_path)
    finally:
        os.remove(temp_file_path)


# LLM-generated content at query #23
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay"}
    result = get_user_config(default_config=custom_config)
    expected = DEFAULT_CONFIG.copy()
    expected.update(custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    custom_config_file = "/tmp/custom_config.yaml"
    result = get_user_config(config_file=custom_config_file)
    assert isinstance(result, dict)

def test_get_user_config_env_config_file():
    os.environ["COOKIECUTTER_CONFIG"] = "/tmp/env_config.yaml"
    result = get_user_config()
    assert isinstance(result, dict)
    del os.environ["COOKIECUTTER_CONFIG"]

def test_get_user_config_user_config_path_exists():
    result = get_user_config()
    assert isinstance(result, dict)

def test_get_user_config_user_config_path_not_exists():
    original_exists = os.path.exists
    os.path.exists = lambda x: False
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    os.path.exists = original_exists


# LLM-generated content at query #24
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    try:
        get_user_config()
    except KeyError:
        assert False, "Predicate at line 40 should evaluate to False when COOKIECUTTER_CONFIG is set"


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.path.exists = lambda path: True if path == USER_CONFIG_PATH else False
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #26
#--------------------------

```
def test_get_config_raises_exception_when_config_path_does_not_exist():
    config_path = '/nonexistent/path/to/config.yaml'
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException to be raised'
    except ConfigDoesNotExistException:
        assert True


# LLM-generated content at query #27
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    config_file = "/custom/config.yaml"
    result = get_user_config(config_file=config_file)
    expected = get_config(config_file)
    assert result == expected

def test_get_user_config_env_config_file(monkeypatch):
    env_config_file = "/env/config.yaml"
    monkeypatch.setenv("COOKIECUTTER_CONFIG", env_config_file)
    result = get_user_config()
    expected = get_config(env_config_file)
    assert result == expected

def test_get_user_config_user_config_path_exists(tmp_path):
    user_config_path = tmp_path / "user_config.yaml"
    user_config_path.write_text("replay_dir: /user/replay")
    result = get_user_config()
    expected = get_config(user_config_path)
    assert result == expected

def test_get_user_config_user_config_path_not_exists():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #28
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    config_file = None
    default_config = False
    USER_CONFIG_PATH = "/some/path"
    try:
        os.environ['COOKIECUTTER_CONFIG']
        assert False, "Predicate at line 40 should evaluate to False"
    except KeyError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_config_file_exists():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #30
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: "~/replays"\ncookiecutters_dir: "~/cookiecutters"')
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent.yaml'
    try:
        get_config(config_path)
        assert False
    except ConfigDoesNotExistException:
        assert True

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w') as f:
        f.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)

def test_get_config_merges_with_defaults():
    config_path = 'partial_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: "~/custom_replays"')
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser('~/custom_replays')
    assert 'cookiecutters_dir' in result  # from defaults
    os.remove(config_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_get_user_config_default_config_true():
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    config = get_user_config(default_config=custom_config)
    expected_config = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected_config

def test_get_user_config_custom_config_file():
    custom_config_file = '/path/to/custom/config.yaml'
    config = get_user_config(config_file=custom_config_file)
    expected_config = get_config(custom_config_file)
    assert config == expected_config

def test_get_user_config_env_config_file():
    os.environ['COOKIECUTTER_CONFIG'] = '/path/to/env/config.yaml'
    config = get_user_config()
    expected_config = get_config('/path/to/env/config.yaml')
    assert config == expected_config
    del os.environ['COOKIECUTTER_CONFIG']

def test_get_user_config_default_user_config():
    config = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        expected_config = get_config(USER_CONFIG_PATH)
        assert config == expected_config
    else:
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #32
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    get_user_config()
    del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    import os
    USER_CONFIG_PATH = "path/to/user/config"
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    assert get_user_config() == get_config(USER_CONFIG_PATH)


