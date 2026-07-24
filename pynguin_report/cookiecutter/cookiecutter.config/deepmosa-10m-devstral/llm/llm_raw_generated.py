####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #4
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

def test_get_config_with_invalid_path():
    config_path = 'nonexistent_config.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #6
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

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


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
def test_expand_path_with_environment_variable():
    os.environ["TEST_VAR"] = "/test/path"
    assert _expand_path("$TEST_VAR") == "/test/path"

def test_expand_path_with_home_directory():
    assert _expand_path("~/test") == os.path.expanduser("~/test")

def test_expand_path_with_both_expansions():
    os.environ["TEST_VAR"] = "test"
    assert _expand_path("~/$TEST_VAR") == os.path.expanduser("~/test")

def test_expand_path_with_no_expansion():
    assert _expand_path("/absolute/path") == "/absolute/path"


# LLM-generated content at query #10
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {'key': 'value'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    with patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_not_set_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_not_set_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #11
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

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_invalid_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'invalid_path'}):
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(ConfigDoesNotExistException):
                get_user_config()


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expandvars(os.path.expanduser('$HOME/replay')),
        'cookiecutters_dir': os.path.expandvars(os.path.expanduser('$HOME/cookiecutters')),
        'other_key': 'value'
    }
    with patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay\ncookiecutters_dir: $HOME/cookiecutters\nother_key: value')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'value'}), \
         patch('config.merge_configs', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'value'}):
        result = get_config(config_path)
        assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with patch('builtins.open', mock_open(read_data='invalid yaml content')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with patch('builtins.open', mock_open(read_data='- list item')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=['list item']):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #14
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
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookiecutters/path',
        'other_key': 'other_value'
    }
    with patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay\ncookiecutters_dir: $HOME/cookiecutters\nother_key: other_value')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookiecutters', 'other_key': 'other_value'}), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x.replace('$HOME', '/expanded')):
        assert get_config(config_path) == expected_config

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


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: ~/test\ncookiecutters_dir: ~/test')

    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser('~/test')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test')
    os.remove(config_path)

def test_get_config_with_nonexistent_path():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_config.yaml')

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid yaml content')

    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('- list item')

    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    os.remove(config_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_config_path_exists_and_is_readable():
    config_path = Path('valid_config.yaml')
    config_path.touch()
    config_path.write_text('key: value', encoding='utf-8')
    assert os.path.exists(config_path)


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
            mock_file.assert_called_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
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

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #21
#--------------------------

```python
def test_config_file_opens_successfully():
    with open(config_path, encoding='utf-8') as file_handle:
        assert True


# LLM-generated content at query #22
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

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = Path('invalid.yaml')
    invalid_yaml_path.write_text('invalid: yaml: content: [', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)
    invalid_yaml_path.unlink()

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = Path('non_dict.yaml')
    non_dict_yaml_path.write_text('- list: item', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)
    non_dict_yaml_path.unlink()

def test_get_config_merges_default_and_yaml_configs():
    yaml_content = {
        'abbreviations': {'custom_abbr': 'value'},
        'replay_dir': '~/custom_replay',
        'cookiecutters_dir': '$HOME/custom_cookies'
    }
    yaml_path = Path('test_config.yaml')
    yaml_path.write_text(yaml.dump(yaml_content), encoding='utf-8')
    config = get_config(yaml_path)
    assert config['abbreviations']['custom_abbr'] == 'value'
    assert config['abbreviations']['default_abbr'] == 'default_value'
    assert config['replay_dir'] == str(Path.home() / 'custom_replay')
    assert config['cookiecutters_dir'] == str(Path.home() / 'custom_cookies')
    yaml_path.unlink()


# LLM-generated content at query #24
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config('path/to/invalid.yaml')


# LLM-generated content at query #25
#--------------------------

```python
def test_yaml_dict_is_instance_of_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #26
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #27
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        },
        'replay_dir': os.path.expandvars(os.path.expanduser('$HOME/.replay')),
        'cookiecutters_dir': os.path.expandvars(os.path.expanduser('$HOME/.cookiecutters'))
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


# LLM-generated content at query #28
#--------------------------

```python
def test_get_config_with_valid_path():
    config_dict = get_config('valid_config.yaml')
    assert isinstance(config_dict, dict)
    assert config_dict['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/replay'))
    assert config_dict['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars('$HOME/cookiecutters'))

def test_get_config_with_nonexistent_path():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_config.yaml')

def test_get_config_with_invalid_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml_config.yaml')

def test_get_config_with_non_dict_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml_config.yaml')


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_yaml_error_raised_when_parsing_invalid_yaml():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(io.StringIO("invalid: yaml: content: [unclosed"))


# LLM-generated content at query #31
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with open('invalid.yaml', 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [[[')
    with raises(InvalidConfiguration):
        get_config('invalid.yaml')


# LLM-generated content at query #32
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml_dict = yaml.safe_load(None) or {}
    assert yaml_dict == {}


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookiecutters/path',
        'other_key': 'other_value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config_with_non_dict_yaml():
    yaml_dict = "not a dictionary"
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #36
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #37
#--------------------------

```python
def test_isinstance_check_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #38
#--------------------------

```python
def test_config_path_does_not_exist():
    assert not os.path.exists('/non/existent/path')


# LLM-generated content at query #39
#--------------------------

```python
def test_get_config_returns_merged_dict_with_expanded_paths():
    config_path = "test_config.yaml"
    yaml_dict = {"replay_dir": "~/test", "cookiecutters_dir": "$HOME/test"}
    expected_config = {
        **DEFAULT_CONFIG,
        "replay_dir": os.path.expanduser("~/test"),
        "cookiecutters_dir": os.path.expandvars("$HOME/test"),
    }

    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(yaml_dict))), \
         patch("os.path.exists", return_value=True), \
         patch("yaml.safe_load", return_value=yaml_dict):
        result = get_config(config_path)

    assert result == expected_config

def test_get_config_raises_exception_if_file_does_not_exist():
    config_path = "nonexistent_config.yaml"

    with patch("os.path.exists", return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_raises_exception_if_yaml_invalid():
    config_path = "invalid_config.yaml"

    with patch("builtins.open", mock_open(read_data="invalid yaml")), \
         patch("os.path.exists", return_value=True), \
         patch("yaml.safe_load", side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_raises_exception_if_yaml_not_dict():
    config_path = "non_dict_config.yaml"
    yaml_list = ["not", "a", "dict"]

    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(yaml_list))), \
         patch("os.path.exists", return_value=True), \
         patch("yaml.safe_load", return_value=yaml_list):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #41
#--------------------------

```python
def test_config_path_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_path.yaml')


# LLM-generated content at query #42
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': 'value2',
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_invalid_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #43
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #44
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': 'value2',
        'replay_dir': '/expanded/path1',
        'cookiecutters_dir': '/expanded/path2'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #45
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookies/path',
        'other_key': 'other_value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #46
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml.safe_load = lambda _: None
    assert not yaml.safe_load('dummy_path')


# LLM-generated content at query #47
#--------------------------

```python
def test_yaml_dict_not_dict_type():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #48
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    config_path = Path('invalid.yaml')
    config_path.write_text('invalid: yaml: content: [', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #49
#--------------------------

```python
def test_get_config_raises_config_does_not_exist_exception():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_path.yaml')

def test_get_config_raises_invalid_configuration_on_parse_error():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml_path.yaml')

def test_get_config_raises_invalid_configuration_on_non_dict_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml_path.yaml')

def test_get_config_returns_merged_config_with_expanded_paths():
    config_path = 'valid_config_path.yaml'
    expected_config = {
        'replay_dir': os.path.expandvars(os.path.expanduser('expected_replay_dir')),
        'cookiecutters_dir': os.path.expandvars(os.path.expanduser('expected_cookies_dir')),
        **DEFAULT_CONFIG
    }
    assert get_config(config_path) == expected_config


# LLM-generated content at query #50
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
    custom_config_file = 'path/to/custom/config'
    result = get_user_config(config_file=custom_config_file)
    assert result == get_config(custom_config_file)

def test_get_user_config_with_env_var_set():
    os.environ['COOKIECUTTER_CONFIG'] = 'path/to/env/config'
    result = get_user_config()
    assert result == get_config('path/to/env/config')

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = True
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = False
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #51
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(open('empty.yaml', encoding='utf-8'))


# LLM-generated content at query #52
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = Path('path/to/existing/config.yaml')
    config_path.touch()
    assert os.path.exists(config_path)


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #55
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookiecutters/path'
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


# LLM-generated content at query #56
#--------------------------

```python
def test_yaml_safe_load_returns_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists('nonexistent_file.yaml')


# LLM-generated content at query #58
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content: [unclosed')):
        with pytest.raises(InvalidConfiguration):
            get_config('dummy_path.yaml')


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #60
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #61
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

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #63
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('test_config.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(open('test_config.yaml', encoding='utf-8'))


# LLM-generated content at query #64
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookiecutters',
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


# LLM-generated content at query #65
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #66
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': 'value2',
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass


# LLM-generated content at query #67
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


# LLM-generated content at query #68
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config(Path("invalid_yaml_file.yaml"))


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    # Ensure the KeyError is raised when 'COOKIECUTTER_CONFIG' is not in os.environ
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #70
#--------------------------

```python
def test_yaml_dict_assignment():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #71
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = Path('path/to/existing/config.yaml')
    config_path.touch()
    assert os.path.exists(config_path)


# LLM-generated content at query #72
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #73
#--------------------------

```python
def test_yaml_safe_load_returns_non_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert yaml_dict is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_with_custom_config_file():
    with patch('builtins.open', mock_open(read_data='replay_dir: /custom/path')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='/custom/config')
            assert result['replay_dir'] == '/custom/path'
            mock_file.assert_called_once_with('/custom/config', encoding='utf-8')

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='replay_dir: /env/path')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result['replay_dir'] == '/env/path'
                mock_file.assert_called_once_with('/env/config', encoding='utf-8')

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='replay_dir: /user/path')) as mock_file:
            result = get_user_config()
            assert result['replay_dir'] == '/user/path'
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #2
#--------------------------

```python
def test_keyerror_raised_when_cookiecutter_config_not_in_environment():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #3
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
    with patch('builtins.open', mock_open(read_data='key: value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #5
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

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #7
#--------------------------

```python
def test_expand_path_with_environment_variable():
    os.environ["TEST_VAR"] = "/test/path"
    assert _expand_path("$TEST_VAR") == "/test/path"

def test_expand_path_with_home_directory():
    assert _expand_path("~/test") == os.path.expanduser("~/test")

def test_expand_path_with_both_expansions():
    os.environ["TEST_VAR"] = "test"
    assert _expand_path("~/$TEST_VAR") == os.path.expanduser("~/test")

def test_expand_path_with_no_expansion_needed():
    assert _expand_path("/absolute/path") == "/absolute/path"


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #10
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

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=False):
            result = get_user_config()
            assert result == DEFAULT_CONFIG


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


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
def test_predicate_at_line_33_evaluates_to_false():
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #17
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

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = 'non_dict_yaml.yaml'
    with open(non_dict_yaml_path, 'w', encoding='utf-8') as f:
        f.write('not_a_dict')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

def test_get_config_merges_default_and_yaml_configs():
    yaml_content = {
        'replay_dir': '~/custom_replay',
        'cookiecutters_dir': '~/custom_cookies',
        'new_key': 'new_value'
    }
    yaml_path = 'test_config.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies')
    assert config['new_key'] == 'new_value'
    assert config['abbreviations'] == DEFAULT_CONFIG['abbreviations']

def test_get_config_expands_environment_variables_in_paths():
    os.environ['TEST_DIR'] = '/test/dir'
    yaml_content = {
        'replay_dir': '$TEST_DIR/replay',
        'cookiecutters_dir': '$TEST_DIR/cookies'
    }
    yaml_path = 'test_config_env.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == '/test/dir/replay'
    assert config['cookiecutters_dir'] == '/test/dir/cookies'


# LLM-generated content at query #18
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_keyerror_raised_when_cookiecutter_config_not_set():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #20
#--------------------------

```python
def test_config_file_predicate_false():
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #21
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
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_without_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #22
#--------------------------

```python
def test_config_path_exists():
    config_path = "existing_config.yaml"
    os.path.exists.return_value = True
    assert os.path.exists(config_path)


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    # Mock os.path.exists to return True for USER_CONFIG_PATH
    import os
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    # Ensure USER_CONFIG_PATH is defined (assuming it's a global variable)
    USER_CONFIG_PATH = "/some/path"
    # Call the function with appropriate arguments to reach line 43
    result = get_user_config()
    # The predicate at line 43 should evaluate to True, leading to line 44-45 execution
    assert result == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #24
#--------------------------

```python
def test_keyerror_predicate_evaluates_to_false():
    os.environ.__delitem__('COOKIECUTTER_CONFIG')
    assert 'COOKIECUTTER_CONFIG' not in os.environ


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

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
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
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #27
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
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file, \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('builtins.open', mock_open(read_data='key: value')) as mock_file, \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #28
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


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
    custom_config_file = 'path/to/custom/config.yaml'
    result = get_user_config(config_file=custom_config_file)
    assert result == get_config(custom_config_file)

def test_get_user_config_with_env_var_set():
    os.environ['COOKIECUTTER_CONFIG'] = 'path/to/env/config.yaml'
    result = get_user_config()
    assert result == get_config('path/to/env/config.yaml')

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = True
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = False
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #31
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
        result = get_user_config(config_file='/custom/path')
        assert result['replay_dir'] == '/test/path'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/path'}), \
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


# LLM-generated content at query #32
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config('invalid_yaml_file.yaml')
    assert 'Unable to parse YAML file' in str(excinfo.value)


# LLM-generated content at query #33
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with patch('builtins.open', mock_open(read_data='')):
        with patch('yaml.safe_load', return_value=None):
            assert not yaml.safe_load(open('dummy_path'))


# LLM-generated content at query #34
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #35
#--------------------------

```python
def test_config_path_exists_and_is_readable():
    config_path = 'path/to/existing/config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('key: value')
    assert os.path.exists(config_path)


# LLM-generated content at query #36
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

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #37
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(f)


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #39
#--------------------------

```python
def test_get_config_returns_merged_default_and_yaml_config():
    yaml_content = """
    key1: value1
    nested:
        key2: value2
    """
    with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file, \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'key1': 'value1', 'nested': {'key2': 'value2'}}), \
         patch('config.merge_configs', return_value={'merged': 'config'}) as mock_merge, \
         patch('config._expand_path', side_effect=lambda x: x):
        result = get_config('dummy_path')
        assert result == {'merged': 'config'}
        mock_merge.assert_called_once_with(DEFAULT_CONFIG, {'key1': 'value1', 'nested': {'key2': 'value2'}})

def test_get_config_raises_exception_if_file_does_not_exist():
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config('non_existent_path')

def test_get_config_raises_exception_if_yaml_invalid():
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config('invalid_yaml_path')

def test_get_config_raises_exception_if_yaml_not_dict():
    with patch('builtins.open', mock_open(read_data='not_a_dict')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value='not_a_dict'):
        with pytest.raises(InvalidConfiguration):
            get_config('non_dict_yaml_path')

def test_get_config_expands_paths_in_config():
    yaml_content = """
    replay_dir: ~/replay
    cookiecutters_dir: $HOME/cookiecutters
    """
    expected_config = {
        'replay_dir': os.path.expanduser('~/replay'),
        'cookiecutters_dir': os.path.expandvars('$HOME/cookiecutters'),
    }
    with patch('builtins.open', mock_open(read_data=yaml_content)), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'replay_dir': '~/replay', 'cookiecutters_dir': '$HOME/cookiecutters'}), \
         patch('config.merge_configs', return_value={'replay_dir': '~/replay', 'cookiecutters_dir': '$HOME/cookiecutters'}), \
         patch('config._expand_path', side_effect=lambda x: os.path.expanduser(os.path.expandvars(x))):
        result = get_config('dummy_path')
        assert result['replay_dir'] == expected_config['replay_dir']
        assert result['cookiecutters_dir'] == expected_config['cookiecutters_dir']


# LLM-generated content at query #40
#--------------------------

```python
def test_open_file_with_encoding():
    config_path = "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("test: value")
    with open(config_path, encoding='utf-8') as file_handle:
        assert file_handle is not None


# LLM-generated content at query #41
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #42
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #43
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #44
#--------------------------

```python
def test_open_file_fails_when_file_does_not_exist():
    config_path = Path('nonexistent_file.yaml')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)


# LLM-generated content at query #45
#--------------------------

```python
def test_config_path_does_not_exist():
    non_existent_path = '/path/to/nonexistent/config.yaml'
    try:
        get_config(non_existent_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #47
#--------------------------

```python
def test_yaml_safe_load_returns_none_or_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #48
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('tests/invalid_yaml.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file tests/invalid_yaml.yaml.'


# LLM-generated content at query #49
#--------------------------

```python
def test_yaml_error_raised_when_parsing_fails():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config("invalid.yaml")
    assert str(exc_info.value) == "Unable to parse YAML file invalid.yaml."


# LLM-generated content at query #50
#--------------------------

```python
def test_get_config_with_valid_path():
    config_dict = get_config('tests/data/config.yaml')
    assert config_dict['replay_dir'] == '/home/user/replay'
    assert config_dict['cookiecutters_dir'] == '/home/user/cookiecutters'
    assert config_dict['default_context'] == {'key': 'value'}

def test_get_config_with_nonexistent_path():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent/path.yaml')

def test_get_config_with_invalid_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('tests/data/invalid.yaml')

def test_get_config_with_non_dict_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('tests/data/non_dict.yaml')


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = "not a dictionary"
    assert not isinstance(yaml_dict, dict)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #2
#--------------------------

```python
def test_expand_path_with_environment_variable():
    os.environ['TEST_VAR'] = '/test/path'
    assert _expand_path('$TEST_VAR') == '/test/path'

def test_expand_path_with_home_directory():
    assert _expand_path('~/test') == os.path.expanduser('~/test')

def test_expand_path_with_both_expansions():
    os.environ['TEST_VAR'] = 'test'
    assert _expand_path('~/$TEST_VAR') == os.path.expanduser('~/test')

def test_expand_path_with_no_expansion():
    assert _expand_path('/absolute/path') == '/absolute/path'

def test_expand_path_with_empty_string():
    assert _expand_path('') == ''


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookiecutters/path',
        'other_key': 'other_value'
    }
    assert get_config(config_path) == expected_config

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


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists.return_value = True
    assert get_user_config() == get_config(USER_CONFIG_PATH)


# LLM-generated content at query #6
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #9
#--------------------------

```python
def test_config_file_predicate_false():
    assert not (None and None is not USER_CONFIG_PATH)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #11
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
    custom_config_path = 'custom_config.yaml'
    with open(custom_config_path, 'w', encoding='utf-8') as f:
        f.write('key: value')
    result = get_user_config(config_file=custom_config_path)
    expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
    assert result == expected
    os.remove(custom_config_path)

def test_get_user_config_with_env_var():
    os.environ['COOKIECUTTER_CONFIG'] = 'env_config.yaml'
    with open('env_config.yaml', 'w', encoding='utf-8') as f:
        f.write('key: value')
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
    assert result == expected
    os.remove('env_config.yaml')
    del os.environ['COOKIECUTTER_CONFIG']

def test_get_user_config_with_user_config_path():
    with open(USER_CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write('key: value')
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
    assert result == expected
    os.remove(USER_CONFIG_PATH)

def test_get_user_config_with_no_config():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #12
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
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                result = get_user_config(config_file='custom_path')
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #13
#--------------------------

```python
def test_config_path_exists():
    config_path = "existing_config.yaml"
    os.path.exists.return_value = True
    assert os.path.exists(config_path) is True


# LLM-generated content at query #14
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

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #15
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #17
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
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_config():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #18
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config_returns_merged_and_expanded_config():
    config_path = 'path/to/config.yaml'
    yaml_dict = {'replay_dir': '$HOME/replays', 'cookiecutters_dir': '~/templates'}
    expected_config = {
        'replay_dir': os.path.expanduser(os.path.expandvars('$HOME/replays')),
        'cookiecutters_dir': os.path.expanduser(os.path.expandvars('~/templates')),
        **DEFAULT_CONFIG
    }
    expected_config.update(yaml_dict)

    with patch('builtins.open', mock_open(read_data=yaml.safe_dump(yaml_dict))), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=yaml_dict):
        result = get_config(config_path)

    assert result == expected_config

def test_get_config_raises_exception_when_file_does_not_exist():
    config_path = 'path/to/nonexistent.yaml'

    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_raises_exception_when_yaml_is_invalid():
    config_path = 'path/to/invalid.yaml'

    with patch('builtins.open', mock_open(read_data='invalid yaml')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    config_path = 'path/to/list.yaml'
    yaml_list = ['not', 'a', 'dict']

    with patch('builtins.open', mock_open(read_data=yaml.safe_dump(yaml_list))), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=yaml_list):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookiecutters'
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


# LLM-generated content at query #22
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        },
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookies/path'
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
    config_path = 'invalid_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    config_path = "nonexistent_file.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)


# LLM-generated content at query #24
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

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #25
#--------------------------

```python
def test_yaml_dict_not_a_dict():
    yaml_dict = "not a dict"
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #26
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = Path('valid_config.yaml')
    config_path.touch()
    assert os.path.exists(config_path)


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists("nonexistent_path")


# LLM-generated content at query #28
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = Path('valid_config.yaml')
    config_path.write_text('replay_dir: ~/replays\ncookiecutters_dir: ~/templates')
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser('~/replays')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/templates')
    assert result['abbreviations'] == DEFAULT_CONFIG['abbreviations']
    config_path.unlink()

def test_get_config_with_nonexistent_path():
    config_path = Path('nonexistent_config.yaml')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = Path('invalid_config.yaml')
    config_path.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    config_path.unlink()

def test_get_config_with_non_dict_yaml():
    config_path = Path('non_dict_config.yaml')
    config_path.write_text('- list item 1\n- list item 2')
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    config_path.unlink()


# LLM-generated content at query #31
#--------------------------

```python
def test_isinstance_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #32
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid_yaml_file.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid_yaml_file.yaml.'


# LLM-generated content at query #33
#--------------------------

```python
def test_yaml_safe_load_raises_yaml_error():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(io.StringIO("invalid yaml content"))


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expanduser(os.path.expandvars('$HOME/replay')),
        'cookiecutters_dir': os.path.expanduser(os.path.expandvars('$HOME/cookiecutters')),
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)


# LLM-generated content at query #35
#--------------------------

```python
def test_yaml_error_raised_when_parsing_invalid_yaml():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("invalid yaml content")


# LLM-generated content at query #36
#--------------------------

```python
def test_get_config_returns_merged_and_expanded_config():
    config_path = 'test_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: ~/test\ncookiecutters_dir: $HOME/test')

    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser('~/test')
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test')
    os.remove(config_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_config_path_exists_and_is_readable():
    config_path = "valid_config.yaml"
    os.path.exists.return_value = True
    open.return_value.__enter__.return_value = "file_content"
    yaml.safe_load.return_value = {"key": "value"}
    merge_configs.return_value = {"replay_dir": "/path", "cookiecutters_dir": "/path"}
    _expand_path.return_value = "/expanded_path"

    result = get_config(config_path)

    assert result == {"replay_dir": "/expanded_path", "cookiecutters_dir": "/expanded_path"}


# LLM-generated content at query #38
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config("invalid_yaml.yaml")
    assert str(exc_info.value) == "Unable to parse YAML file invalid_yaml.yaml."


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists("non_existent_file.yaml")


# LLM-generated content at query #41
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    yaml_dict = {'replay_dir': '~/replays', 'cookiecutters_dir': '$HOME/cookiecutters'}
    DEFAULT_CONFIG = {'replay_dir': 'default_replays', 'cookiecutters_dir': 'default_cookiecutters'}
    expected_config = {'replay_dir': os.path.expanduser('~/replays'), 'cookiecutters_dir': os.path.expandvars('$HOME/cookiecutters')}

    with patch('builtins.open', mock_open(read_data=yaml.safe_dump(yaml_dict))), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=yaml_dict), \
         patch('config.merge_configs', return_value=expected_config):
        result = get_config(config_path)
        assert result == expected_config

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'

    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    yaml_error = yaml.YAMLError('Invalid YAML')

    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', side_effect=yaml_error):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    yaml_dict = ['not', 'a', 'dict']

    with patch('builtins.open', mock_open(read_data=yaml.safe_dump(yaml_dict))), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value=yaml_dict):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #42
#--------------------------

```python
def test_get_config_with_valid_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

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

def test_get_config_with_path_expansion():
    config_path = 'config_with_vars.yaml'
    expected_config = {
        'replay_dir': '/home/user/replay',
        'cookiecutters_dir': '/home/user/cookies',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config


# LLM-generated content at query #43
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

def test_get_config_with_invalid_path():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #44
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(f)


# LLM-generated content at query #45
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = 'valid_config.yaml'
    os.path.exists.return_value = True
    open.return_value.__enter__.return_value = 'file_content'
    yaml.safe_load.return_value = {'replay_dir': '/path', 'cookiecutters_dir': '/path'}
    merge_configs.return_value = {'replay_dir': '/path', 'cookiecutters_dir': '/path'}
    _expand_path.return_value = '/expanded_path'
    result = get_config(config_path)
    assert result == {'replay_dir': '/expanded_path', 'cookiecutters_dir': '/expanded_path'}


# LLM-generated content at query #46
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with patch('builtins.open', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config('valid_path')


# LLM-generated content at query #47
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml.yaml')

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml.yaml')

def test_get_config_merges_default_and_yaml_configs():
    config = get_config('valid_config.yaml')
    assert config == {
        'key1': 'value1',
        'key2': 'value2',
        'nested': {'key3': 'value3'}
    }

def test_get_config_expands_paths():
    config = get_config('valid_config_with_paths.yaml')
    assert config['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/replay'))
    assert config['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))


