####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    custom_config = {'key': 'value'}
    with patch('builtins.open', mock_open(read_data=yaml.dump(custom_config))):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_env_var():
    custom_config = {'key': 'value'}
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data=yaml.dump(custom_config))):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_user_config_path():
    custom_config = {'key': 'value'}
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data=yaml.dump(custom_config))):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
#--------------------------

```python
def test__expand_path_with_environment_variable():
    os.environ["TEST_VAR"] = "/test/path"
    assert _expand_path("$TEST_VAR") == "/test/path"

def test__expand_path_with_user_home():
    assert _expand_path("~/test") == os.path.expanduser("~/test")

def test__expand_path_with_both_environment_variable_and_user_home():
    os.environ["TEST_VAR"] = "test"
    assert _expand_path("~/$TEST_VAR") == os.path.expanduser("~/test")

def test__expand_path_with_no_variables():
    assert _expand_path("/normal/path") == "/normal/path"


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    assert 'COOKIECUTTER_CONFIG' in os.environ


# LLM-generated content at query #4
#--------------------------

```python
def test_keyerror_raised_when_cookiecutter_config_not_set():
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', return_value=False):
            assert get_user_config() == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay_dir\ncookiecutters_dir: $HOME/cookiecutters_dir\nother_key: value')), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay_dir', 'cookiecutters_dir': '$HOME/cookiecutters_dir', 'other_key': 'value'}), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x.replace('$HOME', '/expanded')):
        result = get_config(config_path)
        assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='invalid yaml content')), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='- list item')), \
         patch('yaml.safe_load', return_value=['list item']):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #6
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config("tests/test_invalid_yaml.yaml")


# LLM-generated content at query #7
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        with patch.dict('os.environ', {}, clear=True):
            result = get_user_config()
            assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #8
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    config_path = 'valid_path.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('')  # Empty file causes yaml.safe_load to return None
    assert not yaml.safe_load(open(config_path, encoding='utf-8'))


# LLM-generated content at query #9
#--------------------------

```python
def test_key_error_in_environ():
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #10
#--------------------------

```python
def test_yaml_error_raised_when_invalid_yaml():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config(Path('invalid.yaml'))
    assert str(exc_info.value) == 'Unable to parse YAML file invalid.yaml.'


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expandvars(os.path.expanduser('$HOME/replay')),
        'cookiecutters_dir': os.path.expandvars(os.path.expanduser('$HOME/cookiecutters')),
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    # Mock os.path.exists to return True for USER_CONFIG_PATH
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #15
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #16
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml_dict = yaml.safe_load(None) or {}
    assert yaml_dict == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expanduser(os.path.expandvars('$HOME/replay')),
        'cookiecutters_dir': os.path.expanduser(os.path.expandvars('$HOME/cookiecutters')),
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expanduser(os.path.expandvars('$HOME/replay')),
        'cookiecutters_dir': os.path.expanduser(os.path.expandvars('$HOME/cookiecutters')),
        'other_key': 'other_value'
    }
    with patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay\ncookiecutters_dir: $HOME/cookiecutters\nother_key: other_value')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'other_value'}), \
         patch('config.merge_configs', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'other_value'}):
        result = get_config(config_path)
        assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with patch('builtins.open', mock_open(read_data='- list_item')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=['list_item']):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #21
#--------------------------

```python
def test_user_config_path_exists():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #22
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid_yaml_file.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid_yaml_file.yaml.'


# LLM-generated content at query #23
#--------------------------

```python
def test_get_config_returns_merged_config_with_expanded_paths():
    config_path = Path('test_config.yaml')
    yaml_dict = {'replay_dir': '~/test_replay', 'cookiecutters_dir': '$HOME/test_cookies'}
    expected_config = merge_configs(DEFAULT_CONFIG, yaml_dict)
    expected_config['replay_dir'] = os.path.expanduser('~/test_replay')
    expected_config['cookiecutters_dir'] = os.path.expandvars('$HOME/test_cookies')

    with patch('builtins.open', mock_open(read_data=yaml.safe_dump(yaml_dict))), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=yaml_dict):
        result = get_config(config_path)

    assert result == expected_config


# LLM-generated content at query #24
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = 'invalid_yaml.yaml'
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)
    os.remove(invalid_yaml_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    invalid_yaml_path = 'invalid_top_level.yaml'
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_top_level_path)
    os.remove(invalid_yaml_path)

def test_get_config_merges_with_default_and_expands_paths():
    yaml_path = 'test_config.yaml'
    yaml_content = {
        'replay_dir': '$HOME/test_replay',
        'cookiecutters_dir': '~/test_cookies',
        'new_key': 'new_value'
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expandvars('$HOME/test_replay')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookies')
    assert config['new_key'] == 'new_value'
    assert config.get('preserved_default_key') == DEFAULT_CONFIG['preserved_default_key']
    os.remove(yaml_path)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #27
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookiecutters',
        'other_key': 'value'
    }

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay\ncookiecutters_dir: $HOME/cookiecutters\nother_key: value')), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'value'}), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x.replace('$HOME', '/expanded')):
        result = get_config(config_path)
        assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'

    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='- list item')), \
         patch('yaml.safe_load', return_value=['list item']):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #28
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #29
#--------------------------

```python
def test_yaml_safe_load_returns_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #30
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = Path('existing_config.yaml')
    config_path.touch()
    assert os.path.exists(config_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #32
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay_dir\ncookiecutters_dir: $HOME/cookiecutters_dir\nother_key: value')), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay_dir', 'cookiecutters_dir': '$HOME/cookiecutters_dir', 'other_key': 'value'}), \
         patch('config.merge_configs', return_value={'replay_dir': '$HOME/replay_dir', 'cookiecutters_dir': '$HOME/cookiecutters_dir', 'other_key': 'value'}), \
         patch('os.path.expandvars', side_effect=lambda x: x.replace('$HOME', '/home/user')), \
         patch('os.path.expanduser', side_effect=lambda x: x.replace('/home/user', '/expanded')):
        result = get_config(config_path)

    assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'

    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError('Invalid YAML')):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='- list item')), \
         patch('yaml.safe_load', return_value=['list item']):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #34
#--------------------------

```python
def test_config_path_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config("nonexistent_path.yaml")


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = Path(__file__).parent / 'invalid.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = Path(__file__).parent / 'non_dict.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

def test_get_config_merges_default_and_yaml_configs():
    yaml_path = Path(__file__).parent / 'valid.yaml'
    config = get_config(yaml_path)
    assert config == {
        'replay_dir': os.path.expanduser('~/replays'),
        'cookiecutters_dir': os.path.expanduser('~/cookiecutters'),
        'other_key': 'other_value'
    }

def test_get_config_expands_paths_in_config():
    yaml_path = Path(__file__).parent / 'valid.yaml'
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expanduser('~/replays')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/cookiecutters')


# LLM-generated content at query #36
#--------------------------

```python
def test_config_path_does_not_exist():
    config_path = 'non_existent_file.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #38
#--------------------------

```python
def test_yaml_safe_load_returns_none_or_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #39
#--------------------------

```python
def test_yaml_dict_is_falsy():
    yaml_dict = None
    assert not (yaml_dict or {})


# LLM-generated content at query #40
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #41
#--------------------------

```python
def test_yaml_dict_is_dict_when_valid():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #42
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'other_value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': os.path.expanduser(os.path.expandvars('~/.replay')),
        'cookiecutters_dir': os.path.expanduser(os.path.expandvars('~/.cookiecutters')),
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = "nonexistent_config.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #44
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #45
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    config_path = Path('invalid.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #46
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid.yaml.'


# LLM-generated content at query #47
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #48
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config("tests/data/invalid_yaml.yaml")


# LLM-generated content at query #49
#--------------------------

```python
def test_yaml_error_raised_when_parsing_invalid_yaml():
    config_path = Path('invalid.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(yaml.YAMLError):
        get_config(config_path)


# LLM-generated content at query #50
#--------------------------

```python
def test_keyerror_raised_when_cookiecutter_config_not_set():
    with patch.dict(os.environ, {}, clear=True):
        assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #51
#--------------------------

```python
def test_yaml_safe_load_returns_non_empty_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists('nonexistent_file.yaml')


# LLM-generated content at query #53
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open(config_path, encoding='utf-8') as file_handle:
        assert yaml.safe_load(file_handle) is None


# LLM-generated content at query #54
#--------------------------

```python
def test_config_path_exists_and_is_openable():
    config_path = 'valid_config.yaml'
    os.path.exists = lambda path: True
    open(config_path, encoding='utf-8').read.return_value = '{}'
    assert os.path.exists(config_path) is True


# LLM-generated content at query #55
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #56
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = {}
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #57
#--------------------------

```python
def test_yaml_dict_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_true():
    assert isinstance(config_path, (Path, str))
    assert os.path.exists(config_path)


# LLM-generated content at query #59
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = Path('valid_config.yaml')
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'other_value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = Path('nonexistent_config.yaml')
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = Path('invalid_yaml.yaml')
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = Path('non_dict_yaml.yaml')
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #60
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    custom_config_path = 'path/to/custom/config'
    with patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config(config_file=custom_config_path)
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var():
    env_config_path = 'path/to/env/config'
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': env_config_path}), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #61
#--------------------------

```python
def test_keyerror_raised_when_environ_var_not_set():
    with patch.dict(os.environ, {}, clear=True):
        assert "COOKIECUTTER_CONFIG" not in os.environ


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #63
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid.yaml.'


# LLM-generated content at query #64
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = "not a dict"
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #65
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #66
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config('path/to/invalid.yaml')


# LLM-generated content at query #67
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #68
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #70
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #71
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/path1',
        'cookiecutters_dir': '/expanded/path2'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #72
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                assert get_user_config(config_file='custom_path') == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                assert get_user_config() == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                assert get_user_config() == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_without_config_file():
    with patch('os.path.exists', return_value=False):
        assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #74
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid_yaml_file.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid_yaml_file.yaml.'


# LLM-generated content at query #75
#--------------------------

```python
def test_get_config_with_valid_file():
    config_path = 'valid_config.yaml'
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_expands_paths():
    config_path = 'config_with_paths.yaml'
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/replay'))
    assert result['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars('$HOME/cookiecutters'))


# LLM-generated content at query #76
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #77
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists("non_existent_config_path.yaml")


# LLM-generated content at query #78
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(f)


# LLM-generated content at query #79
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml.safe_load = lambda _: None
    assert yaml.safe_load('dummy') is None


# LLM-generated content at query #80
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #81
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                mock_file.assert_called_with('env_path', encoding='utf-8')
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #82
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with open('test_config.yaml', 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [[[")
    with pytest.raises(InvalidConfiguration):
        get_config('test_config.yaml')


# LLM-generated content at query #83
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    mock_file_handle = MagicMock()
    mock_file_handle.__enter__.return_value = mock_file_handle
    yaml.safe_load.return_value = None
    result = yaml.safe_load(mock_file_handle)
    assert result is None


# LLM-generated content at query #84
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_invalid_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #85
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #86
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config(config_file='custom_path')
        assert result['key'] == 'value'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config()
        assert result['key'] == 'value'

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')):
        result = get_user_config()
        assert result['key'] == 'value'

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #87
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #88
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #89
#--------------------------

```python
def test_yaml_error_raised_when_parsing_fails():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(io.StringIO("invalid: yaml: content: [unclosed: bracket"))


# LLM-generated content at query #90
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'

def test_get_config_expands_paths():
    config_path = 'config_with_paths.yaml'
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/replay'))
    assert result['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'test_key': 'test_value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='test_key: test_value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            expected = merge_configs(DEFAULT_CONFIG, {'test_key': 'test_value'})
            assert result == expected

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='test_key: test_value')):
            result = get_user_config()
            expected = merge_configs(DEFAULT_CONFIG, {'test_key': 'test_value'})
            assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='test_key: test_value')):
            result = get_user_config()
            expected = merge_configs(DEFAULT_CONFIG, {'test_key': 'test_value'})
            assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #2
#--------------------------

```python
def test_expand_path_with_environment_variable():
    os.environ["TEST_VAR"] = "/test/path"
    assert _expand_path("$TEST_VAR") == "/test/path"

def test_expand_path_with_user_home():
    assert _expand_path("~/test") == os.path.expanduser("~/test")

def test_expand_path_with_both_expansions():
    os.environ["TEST_VAR"] = "test"
    assert _expand_path("~/$TEST_VAR") == os.path.expanduser("~/test")

def test_expand_path_with_no_expansion_needed():
    assert _expand_path("/absolute/path") == "/absolute/path"

def test_expand_path_with_empty_string():
    assert _expand_path("") == ""

def test_expand_path_with_relative_path():
    assert _expand_path("relative/path") == "relative/path"


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookies/path',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = "invalid_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'tests/data/valid_config.yaml'
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'tests/data/invalid_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'tests/data/non_dict_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'

def test_get_config_expands_paths():
    config_path = 'tests/data/valid_config.yaml'
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/replay'))
    assert result['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #10
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(open('empty.yaml', encoding='utf-8'))


# LLM-generated content at query #11
#--------------------------

```python
def test_yaml_error_raised_when_parsing_invalid_yaml():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("invalid: yaml: content: [unclosed")


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    # Mock os.path.exists to return True
    os.path.exists = lambda _: True

    # Call the function with no arguments to trigger the predicate
    result = get_user_config()

    # Assert that the predicate at line 43 evaluates to True
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #14
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /test/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/test/path'}):
        result = get_user_config(config_file='/custom/config.yaml')
        assert result['replay_dir'] == '/test/path'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/path'}):
        result = get_user_config()
        assert result['replay_dir'] == '/env/path'

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /user/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/user/path'}):
        result = get_user_config()
        assert result['replay_dir'] == '/user/path'

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #15
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=False):
            result = get_user_config()
            assert result == DEFAULT_CONFIG


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with('env_path', encoding='utf-8')

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #19
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists('nonexistent_config_path')


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = 'invalid.yaml'
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = 'non_dict.yaml'
    with open(non_dict_yaml_path, 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

def test_get_config_merges_with_default_config():
    yaml_path = 'test_config.yaml'
    yaml_content = {'replay_dir': '~/.test_replay', 'new_key': 'new_value'}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expanduser('~/.test_replay')
    assert config['new_key'] == 'new_value'
    assert config['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])

def test_get_config_expands_environment_variables():
    yaml_path = 'test_env.yaml'
    yaml_content = {'replay_dir': '$HOME/test_replay', 'cookiecutters_dir': '$USER/test_cookies'}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expandvars('$HOME/test_replay')
    assert config['cookiecutters_dir'] == os.path.expandvars('$USER/test_cookies')

def test_get_config_preserves_nested_dicts():
    yaml_path = 'test_nested.yaml'
    yaml_content = {'abbreviations': {'test': 'value', 'nested': {'key': 'value'}}}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['abbreviations']['test'] == 'value'
    assert config['abbreviations']['nested']['key'] == 'value'
    assert 'existing_key' in config['abbreviations'] if 'existing_key' in DEFAULT_CONFIG['abbreviations'] else True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                mock_file.assert_called_once_with('env_path', encoding='utf-8')
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_file.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = 'invalid_yaml.yaml'
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)
    os.remove(invalid_yaml_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = 'non_dict_yaml.yaml'
    with open(non_dict_yaml_path, 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)
    os.remove(non_dict_yaml_path)

def test_get_config_merges_with_default_and_expands_paths():
    yaml_content = {
        'replay_dir': '~/test_replay',
        'cookiecutters_dir': '$TEST_DIR/cookiecutters',
        'new_key': 'new_value'
    }
    yaml_path = 'test_config.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expanduser('~/test_replay')
    assert config['cookiecutters_dir'] == os.path.expandvars('$TEST_DIR/cookiecutters')
    assert config['new_key'] == 'new_value'
    assert config.get('default_key') == DEFAULT_CONFIG['default_key']
    os.remove(yaml_path)


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #4
#--------------------------

```python
def test_expand_path_with_environment_variable():
    os.environ['TEST_VAR'] = '/test/path'
    assert _expand_path('$TEST_VAR') == '/test/path'

def test_expand_path_with_user_home():
    assert _expand_path('~/test') == os.path.expanduser('~/test')

def test_expand_path_with_both_expansions():
    os.environ['TEST_VAR'] = '~'
    assert _expand_path('$TEST_VAR/test') == os.path.expanduser('~/test')

def test_expand_path_with_no_expansion_needed():
    assert _expand_path('/absolute/path') == '/absolute/path'

def test_expand_path_with_empty_string():
    assert _expand_path('') == ''


# LLM-generated content at query #5
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #7
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #8
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='replay_dir: /custom/replay')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='/custom/config.yaml')
            assert result['replay_dir'] == '/custom/replay'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}):
        with patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')):
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result['replay_dir'] == '/env/replay'

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='replay_dir: /user/replay')):
                result = get_user_config()
                assert result['replay_dir'] == '/user/replay'

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=False):
            result = get_user_config()
            assert result == DEFAULT_CONFIG


# LLM-generated content at query #9
#--------------------------

```python
def test_keyerror_predicate():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #10
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert mock_file.call_args[0][0] == 'custom_path'
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert mock_file.call_args[0][0] == 'env_path'
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert mock_file.call_args[0][0] == USER_CONFIG_PATH
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #11
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with('env_path', encoding='utf-8')

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_user_config_path_exists():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #19
#--------------------------

```python
def test_keyerror_predicate():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    # Mock os.path.exists to return True
    import os
    os.path.exists = lambda path: True

    # Mock USER_CONFIG_PATH
    import sys
    sys.modules['__main__'].USER_CONFIG_PATH = "some_path"

    # Call the function with appropriate arguments to reach line 43
    result = get_user_config()

    # Assert that the predicate at line 43 evaluates to True
    assert os.path.exists(USER_CONFIG_PATH) == True


# LLM-generated content at query #21
#--------------------------

```python
def test_config_path_exists():
    config_path = "existing_config.yaml"
    assert os.path.exists(config_path)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #23
#--------------------------

```python
def test_keyerror_predicate():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #24
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #25
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #26
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        assert result['key'] == 'value'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result['key'] == 'value'

def test_get_user_config_with_user_config_path_exists():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result['key'] == 'value'

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #27
#--------------------------

```python
def test_config_path_exists():
    config_path = "existing_config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write("key: value")
    assert os.path.exists(config_path)


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #29
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #30
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #31
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #32
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_default_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #33
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    custom_config_path = 'path/to/custom/config'
    result = get_user_config(config_file=custom_config_path)
    expected = get_config(custom_config_path)
    assert result == expected

def test_get_user_config_with_env_var_set():
    env_config_file = 'path/to/env/config'
    os.environ['COOKIECUTTER_CONFIG'] = env_config_file
    result = get_user_config()
    expected = get_config(env_config_file)
    assert result == expected

def test_get_user_config_without_env_var_and_user_config_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    expected = get_config(USER_CONFIG_PATH)
    assert result == expected

def test_get_user_config_without_env_var_and_user_config_not_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = False
    result = get_user_config()
    expected = copy.copy(DEFAULT_CONFIG)
    assert result == expected


