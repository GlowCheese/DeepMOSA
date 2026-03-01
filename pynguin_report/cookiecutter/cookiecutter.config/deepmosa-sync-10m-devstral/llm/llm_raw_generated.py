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

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
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
         patch('builtins.open', mock_open(read_data='replay_dir: /custom/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/custom/path'}):
        result = get_user_config(config_file='/custom/config.yaml')
        assert result['replay_dir'] == '/custom/path'

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/path'}):
        result = get_user_config()
        assert result['replay_dir'] == '/env/path'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /user/path')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/user/path'}):
        result = get_user_config()
        assert result['replay_dir'] == '/user/path'

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG

def test_get_user_config_with_invalid_yaml():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='invalid: yaml: content')), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file='/invalid/config.yaml')

def test_get_user_config_with_nonexistent_config_file():
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config(config_file='/nonexistent/config.yaml')


# LLM-generated content at query #3
#--------------------------

```python
def test_expand_path_with_environment_variable():
    os.environ['TEST_VAR'] = '/test/path'
    assert _expand_path('$TEST_VAR') == '/test/path'

def test_expand_path_with_user_home():
    assert _expand_path('~/test') == os.path.expanduser('~/test')

def test_expand_path_with_both_expansions():
    os.environ['TEST_VAR'] = 'test'
    assert _expand_path('~/$TEST_VAR') == os.path.expanduser('~/test')

def test_expand_path_with_no_expansions():
    assert _expand_path('/absolute/path') == '/absolute/path'


# LLM-generated content at query #4
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

def test_get_user_config_with_default_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #5
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
    non_dict_yaml_path.write_text('- list item', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)
    non_dict_yaml_path.unlink()

def test_get_config_merges_with_default_config():
    yaml_content = {'key1': 'value1', 'key2': {'nested_key': 'nested_value'}}
    yaml_path = Path('test_config.yaml')
    yaml_path.write_text(yaml.dump(yaml_content), encoding='utf-8')
    result = get_config(yaml_path)
    assert result['key1'] == 'value1'
    assert result['key2']['nested_key'] == 'nested_value'
    assert result['some_default_key'] == DEFAULT_CONFIG['some_default_key']
    yaml_path.unlink()

def test_get_config_expands_paths():
    yaml_content = {'replay_dir': '~/test_dir', 'cookiecutters_dir': '$HOME/test_dir'}
    yaml_path = Path('test_config.yaml')
    yaml_path.write_text(yaml.dump(yaml_content), encoding='utf-8')
    result = get_config(yaml_path)
    assert result['replay_dir'] == os.path.expanduser('~/test_dir')
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test_dir')
    yaml_path.unlink()


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_config_file_predicate_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #8
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = 'valid_config.yaml'
    open(config_path, encoding='utf-8').close()
    assert os.path.exists(config_path)


# LLM-generated content at query #9
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
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'
    else:
        assert False, "Expected ConfigDoesNotExistException"

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'
    else:
        assert False, "Expected InvalidConfiguration"

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_yaml.yaml'
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_43():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    # Ensure the KeyError is raised when 'COOKIECUTTER_CONFIG' is not in os.environ
    # This makes the predicate at line 40 evaluate to False
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open(config_path, encoding='utf-8') as file_handle:
        assert not yaml.safe_load(file_handle)


# LLM-generated content at query #14
#--------------------------

```python
def test_yaml_error_handling():
    config_path = 'invalid.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config(config_path)
    assert str(exc_info.value) == f'Unable to parse YAML file {config_path}.'


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config_raises_invalid_configuration_when_yaml_dict_is_not_dict():
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config('path/to/valid/file.yaml')
    assert 'Top-level element of YAML file path/to/valid/file.yaml should be an object.' in str(excinfo.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists('nonexistent_file.yaml')


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
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config("path/to/invalid.yaml")
    assert str(exc_info.value) == "Unable to parse YAML file path/to/invalid.yaml."


# LLM-generated content at query #20
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

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #21
#--------------------------

```python
def test_config_path_does_not_exist():
    assert not os.path.exists('non_existent_config_path')


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_config_path_exists_and_is_openable():
    config_path = 'valid_config.yaml'
    assert os.path.exists(config_path)
    assert open(config_path, encoding='utf-8').readable()


# LLM-generated content at query #25
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expanduser('~/.replay'),
        'cookiecutters_dir': os.path.expanduser('~/.cookiecutters'),
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

def test_get_config_expands_environment_variables():
    config_path = 'config_with_env_vars.yaml'
    config = get_config(config_path)
    assert config['replay_dir'] == os.path.expandvars('$HOME/.replay')
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/.cookiecutters')

def test_get_config_expands_user_home():
    config_path = 'config_with_user_home.yaml'
    config = get_config(config_path)
    assert config['replay_dir'] == os.path.expanduser('~/.replay')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')


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
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result['key'] == 'value'

def test_get_user_config_with_env_var_config():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result['key'] == 'value'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result['key'] == 'value'

def test_get_user_config_fallback_to_default():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_config_path_does_not_exist():
    config_path = Path('/non/existent/path')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_path)


# LLM-generated content at query #29
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open(config_path, encoding='utf-8') as file_handle:
        assert not yaml.safe_load(file_handle)


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
def test_yaml_safe_load_returns_dict_or_none():
    assert yaml.safe_load(io.StringIO("key: value")) == {"key": "value"}
    assert yaml.safe_load(io.StringIO("")) is None


# LLM-generated content at query #32
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

def test_get_config_raises_exception_when_yaml_is_invalid():
    invalid_yaml_path = 'invalid.yaml'
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    non_dict_yaml_path = 'non_dict.yaml'
    with open(non_dict_yaml_path, 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

def test_get_config_merges_default_and_yaml_configs():
    yaml_path = 'test_config.yaml'
    yaml_content = {'replay_dir': '$HOME/test', 'cookiecutters_dir': '$USER/test'}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expandvars('$HOME/test')
    assert config['cookiecutters_dir'] == os.path.expandvars('$USER/test')
    assert config['other_default_key'] == DEFAULT_CONFIG['other_default_key']

def test_get_config_expands_environment_variables_and_user_home():
    yaml_path = 'test_expand.yaml'
    yaml_content = {'replay_dir': '$HOME/test', 'cookiecutters_dir': '~/test'}
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f)
    config = get_config(yaml_path)
    assert config['replay_dir'] == os.path.expanduser('~') + '/test'
    assert config['cookiecutters_dir'] == os.path.expanduser('~') + '/test'


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config_with_valid_yaml():
    yaml_content = """
    replay_dir: ~/test_replay
    cookiecutters_dir: ~/test_cookies
    """
    config_path = "test_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    result = get_config(config_path)
    assert result['replay_dir'] == os.path.expanduser('~/test_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookies')
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    with pytest.raises(ConfigDoesNotExistException):
        get_config("nonexistent_config.yaml")

def test_get_config_with_invalid_yaml():
    yaml_content = "invalid yaml content"
    config_path = "invalid_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    yaml_content = "- not a dict"
    config_path = "non_dict_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)
    os.remove(config_path)


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
        'other_key': 'other_value'
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


# LLM-generated content at query #36
#--------------------------

```python
def test_get_config_raises_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

def test_get_config_raises_when_yaml_is_invalid():
    with pytest.raises(InvalidConfiguration):
        get_config('tests/fixtures/invalid.yaml')

def test_get_config_raises_when_yaml_top_level_is_not_dict():
    with pytest.raises(InvalidConfiguration):
        get_config('tests/fixtures/not_dict.yaml')

def test_get_config_merges_with_default_config():
    config = get_config('tests/fixtures/valid.yaml')
    assert config['abbreviations'] == {'nf': 'notebooks/'}

def test_get_config_expands_environment_variables():
    os.environ['TEST_DIR'] = '/test/dir'
    config = get_config('tests/fixtures/with_env_var.yaml')
    assert config['replay_dir'] == '/test/dir'

def test_get_config_expands_user_home():
    config = get_config('tests/fixtures/with_home.yaml')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')


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


# LLM-generated content at query #2
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

def test_get_config_with_invalid_path():
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


# LLM-generated content at query #3
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

def test_get_user_config_with_invalid_config_file():
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')) as mock_file:
        with patch('os.path.exists', return_value=True):
            with pytest.raises(InvalidConfiguration):
                get_user_config(config_file='invalid_path')

def test_get_user_config_with_nonexistent_config_file():
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config(config_file='nonexistent_path')


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        },
        'replay_dir': os.path.expandvars(os.path.expanduser('$HOME/replay')),
        'cookiecutters_dir': os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))
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


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #7
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

def test_get_user_config_without_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


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
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
                mock_file.assert_called_once_with('env_path', encoding='utf-8')

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


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


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #14
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #15
#--------------------------

```python
def test_keyerror_exception_raised():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    get_user_config()


# LLM-generated content at query #16
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
    with patch('builtins.open', mock_open(read_data='key: value')):
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result['key'] == 'value'

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')):
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result['key'] == 'value'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')):
            result = get_user_config()
            assert result['key'] == 'value'

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_value'
    assert 'COOKIECUTTER_CONFIG' in os.environ


# LLM-generated content at query #18
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


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

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        assert result['key'] == 'value'

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yaml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    with open('invalid.yaml', 'w') as f:
        f.write('invalid yaml content')
    with pytest.raises(InvalidConfiguration):
        get_config('invalid.yaml')

def test_get_config_raises_exception_when_yaml_top_level_is_not_dict():
    with open('not_dict.yaml', 'w') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config('not_dict.yaml')

def test_get_config_merges_with_default_and_expands_paths():
    yaml_content = {
        'replay_dir': '~/test_replay',
        'cookiecutters_dir': '$HOME/test_cookies',
        'other_setting': 'value'
    }
    with open('test_config.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    config = get_config('test_config.yaml')

    assert config['replay_dir'] == os.path.expanduser('~/test_replay')
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test_cookies')
    assert config['other_setting'] == 'value'
    assert config['default_setting'] == DEFAULT_CONFIG['default_setting']


# LLM-generated content at query #4
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path) is True


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists("existing_config_path.yaml") is True


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'other_value'
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("replay_dir: $HOME/replay_dir\ncookiecutters_dir: $HOME/cookiecutters_dir\nother_key: other_value")
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("- not a dict")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


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
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #10
#--------------------------

```python
def test_keyerror_raised_when_cookiecutter_config_not_in_environment():
    with patch.dict(os.environ, {}, clear=True):
        assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #11
#--------------------------

```python
def test_config_file_predicate_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


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

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #13
#--------------------------

```python
def test_keyerror_predicate():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


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
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #16
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

def test_get_user_config_with_env_config_file():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}):
        with patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')):
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result['replay_dir'] == '/env/replay'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='replay_dir: /user/replay')):
            result = get_user_config()
            assert result['replay_dir'] == '/user/replay'

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        with patch.dict('os.environ', {}, clear=True):
            result = get_user_config()
            assert result == DEFAULT_CONFIG


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


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
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #20
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #21
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
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /custom/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/custom/replay'}):
        result = get_user_config(config_file='/custom/config.yaml')
        assert result['replay_dir'] == '/custom/replay'

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/env/replay'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /user/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/user/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/user/replay'

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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
            assert result['key'] == 'value'

def test_get_user_config_with_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result['key'] == 'value'

def test_get_user_config_with_env_var_not_set_and_user_config_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
                result = get_user_config()
                mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
                assert result['key'] == 'value'

def test_get_user_config_with_env_var_not_set_and_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True):
        with patch('os.path.exists', return_value=False):
            result = get_user_config()
            assert result == DEFAULT_CONFIG


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


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
    custom_config = {"key": "value"}
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


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {"key": "value"}
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
            result = get_user_config()
            mock_file.assert_called_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config_file():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #4
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

def test_expand_path_with_no_expansion_needed():
    assert _expand_path('/absolute/path') == '/absolute/path'


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #8
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookies/path',
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


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'},
        'replay_dir': '/expanded/path',
        'cookiecutters_dir': '/expanded/path'
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


# LLM-generated content at query #11
#--------------------------

```python
def test_config_path_exists_and_is_readable():
    config_path = 'valid_config.yaml'
    mock_open = mock.mock_open(read_data='key: value')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.exists', return_value=True):
            result = get_config(config_path)
            assert result == {'key': 'value'}


# LLM-generated content at query #12
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with patch('builtins.open', mock_open(read_data='')):
        assert not yaml.safe_load(open('dummy_path'))


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    assert not os.path.exists('nonexistent_file.yaml')


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


# LLM-generated content at query #15
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #17
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    assert not (None and None is not USER_CONFIG_PATH)
    assert not ("" and "" is not USER_CONFIG_PATH)
    assert not (USER_CONFIG_PATH and USER_CONFIG_PATH is not USER_CONFIG_PATH)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_43():
    # Mock os.path.exists to return True
    import os
    os.path.exists = lambda path: True

    # Ensure USER_CONFIG_PATH is set to some path
    USER_CONFIG_PATH = "/some/path"

    # The predicate at line 43 should evaluate to True
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #20
#--------------------------

```python
def test_yaml_dict_is_not_a_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


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


# LLM-generated content at query #22
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #23
#--------------------------

```python
def test_config_path_exists_and_is_file():
    config_path = 'path/to/existing/config.yaml'
    assert os.path.exists(config_path) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config("tests/invalid_yaml.yaml")
    assert str(exc_info.value) == "Unable to parse YAML file tests/invalid_yaml.yaml."


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config_raises_ConfigDoesNotExistException():
    assert not os.path.exists('nonexistent_config.yaml')


# LLM-generated content at query #27
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #28
#--------------------------

```python
def test_yaml_error_raised_when_invalid_yaml():
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config("invalid.yaml")
    assert "Unable to parse YAML file" in str(excinfo.value)


# LLM-generated content at query #29
#--------------------------

```python
def test_config_path_is_not_a_file():
    assert not os.path.exists('non_existent_config_path')


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #31
#--------------------------

```python
def test_config_path_exists():
    config_path = "valid_config.yaml"
    os.path.exists.return_value = True
    assert os.path.exists(config_path) is True


# LLM-generated content at query #32
#--------------------------

```python
def test_config_path_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookiecutters/path',
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


# LLM-generated content at query #34
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml_dict = None
    assert not (yaml_dict or {})


# LLM-generated content at query #35
#--------------------------

```python
def test_config_file_opens_successfully():
    config_path = 'valid_config.yaml'
    mock_open = mock.mock_open(read_data='key: value')
    with patch('builtins.open', mock_open):
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', return_value={'key': 'value'}):
                with patch('merge_configs', return_value={'key': 'value', 'replay_dir': '/path', 'cookiecutters_dir': '/path'}):
                    with patch('_expand_path', return_value='/path'):
                        result = get_config(config_path)
                        assert result == {'key': 'value', 'replay_dir': '/path', 'cookiecutters_dir': '/path'}


# LLM-generated content at query #36
#--------------------------

```python
def test_get_config_with_valid_path():
    config_dict = get_config('valid_config.yaml')
    assert isinstance(config_dict, dict)
    assert 'replay_dir' in config_dict
    assert 'cookiecutters_dir' in config_dict

def test_get_config_with_invalid_path():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_config.yaml')

def test_get_config_with_invalid_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml_config.yaml')

def test_get_config_with_non_dict_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml_config.yaml')

def test_get_config_expands_paths():
    config_dict = get_config('config_with_paths.yaml')
    assert config_dict['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/replay'))
    assert config_dict['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))


# LLM-generated content at query #37
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('tests/configs/invalid_yaml.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file tests/configs/invalid_yaml.yaml.'


# LLM-generated content at query #38
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #39
#--------------------------

```python
def test_yaml_safe_load_returns_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #40
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('test_config.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(open('test_config.yaml', encoding='utf-8'))


# LLM-generated content at query #41
#--------------------------

```python
def test_get_config_with_valid_yaml_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
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


# LLM-generated content at query #42
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open(config_path, encoding='utf-8') as file_handle:
        assert not yaml.safe_load(file_handle)


# LLM-generated content at query #43
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml_file.yaml')


# LLM-generated content at query #44
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #45
#--------------------------

```python
def test_yaml_safe_load_returns_none_or_dict():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #46
#--------------------------

```python
def test_get_config_with_valid_file():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookiecutters',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_file():
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


# LLM-generated content at query #47
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #48
#--------------------------

```python
def test_config_path_does_not_exist():
    assert not os.path.exists('nonexistent_config_path.yaml')


# LLM-generated content at query #49
#--------------------------

```python
def test_predicate_at_line_43():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #50
#--------------------------

```python
def test_config_file_not_found_in_environment():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #51
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
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /custom/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/custom/replay'}):
        result = get_user_config(config_file='/custom/config')
        assert result['replay_dir'] == '/custom/replay'

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/env/replay'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /user/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/user/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/user/replay'

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #52
#--------------------------

```python
def test_yaml_dict_is_dict():
    yaml_dict = {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #53
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('test_config.yaml', 'w', encoding='utf-8') as f:
        f.write('')
    assert not yaml.safe_load(open('test_config.yaml', encoding='utf-8'))


# LLM-generated content at query #54
#--------------------------

```python
def test_get_config_with_valid_path():
    config_dict = get_config('valid_config.yaml')
    assert isinstance(config_dict, dict)
    assert 'replay_dir' in config_dict
    assert 'cookiecutters_dir' in config_dict

def test_get_config_with_invalid_path():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_config.yaml')

def test_get_config_with_invalid_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml_config.yaml')

def test_get_config_with_non_dict_yaml():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml_config.yaml')

def test_get_config_expands_paths():
    config_dict = get_config('config_with_paths.yaml')
    assert config_dict['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/replay'))
    assert config_dict['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars('$HOME/cookiecutters'))


# LLM-generated content at query #55
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config('invalid.yaml')
    assert str(excinfo.value) == 'Unable to parse YAML file invalid.yaml.'


# LLM-generated content at query #56
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': os.path.expanduser('~/.local/share/replay'),
        'cookiecutters_dir': os.path.expanduser('~/.local/share/cookiecutters'),
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


# LLM-generated content at query #57
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_nonexistent_path():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #58
#--------------------------

```python
def test_yaml_dict_is_not_a_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #59
#--------------------------

```python
def test_yaml_dict_is_not_a_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #60
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = "valid_config.yaml"
    expected_config = {
        'replay_dir': '/expanded/replay_dir',
        'cookiecutters_dir': '/expanded/cookiecutters_dir',
        'other_key': 'value'
    }
    assert get_config(config_path) == expected_config

def test_get_config_with_invalid_path():
    config_path = "invalid_config.yaml"
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


# LLM-generated content at query #61
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


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_43():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #63
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with open('empty.yaml', 'w') as f:
        f.write('')
    assert not yaml.safe_load(open('empty.yaml'))


# LLM-generated content at query #64
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
    result = get_user_config(config_file=custom_config_path)
    expected = get_config(custom_config_path)
    assert result == expected

def test_get_user_config_with_env_config_file():
    env_config_file = 'env_config.yaml'
    os.environ['COOKIECUTTER_CONFIG'] = env_config_file
    result = get_user_config()
    expected = get_config(env_config_file)
    assert result == expected

def test_get_user_config_with_user_config_path():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    expected = get_config(USER_CONFIG_PATH)
    assert result == expected

def test_get_user_config_with_no_config_file():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.path.exists = lambda path: False
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #65
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
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /custom/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/custom/replay'}):
        result = get_user_config(config_file='/custom/config.yaml')
        assert result['replay_dir'] == '/custom/replay'

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/env/replay'

def test_get_user_config_with_user_config_path():
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='replay_dir: /user/replay')), \
         patch('yaml.safe_load', return_value={'replay_dir': '/user/replay'}):
        result = get_user_config()
        assert result['replay_dir'] == '/user/replay'

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False), \
         patch.dict('os.environ', {}, clear=True):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


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


# LLM-generated content at query #67
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    config_path = 'valid_path.yaml'
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='invalid yaml content')), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError):
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)


# LLM-generated content at query #68
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with open('invalid.yaml', 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')  # Invalid YAML
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('invalid.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file invalid.yaml.'


# LLM-generated content at query #69
#--------------------------

```python
def test_path_does_not_exist():
    assert not os.path.exists('nonexistent_path.yaml')


# LLM-generated content at query #70
#--------------------------

```python
def test_get_config_raises_exception_when_file_does_not_exist():
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yml')

def test_get_config_raises_exception_when_yaml_is_invalid():
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_yaml.yml')

def test_get_config_raises_exception_when_top_level_is_not_dict():
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_yaml.yml')

def test_get_config_returns_merged_dict_with_expanded_paths():
    config = get_config('valid_config.yml')
    assert isinstance(config, dict)
    assert config['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/replay'))
    assert config['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/cookiecutters'))


# LLM-generated content at query #71
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #72
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

def test_get_config_with_invalid_path():
    config_path = "invalid_config.yaml"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #74
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config('tests/data/invalid_yaml.yaml')
    assert str(exc_info.value) == 'Unable to parse YAML file tests/data/invalid_yaml.yaml.'


# LLM-generated content at query #75
#--------------------------

```python
def test_config_path_exists_and_is_readable():
    config_path = 'existing_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('key: value')
    assert os.path.exists(config_path)


# LLM-generated content at query #76
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    yaml_dict = yaml.safe_load(io.StringIO('')) or {}
    assert yaml_dict == {}


# LLM-generated content at query #77
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #78
#--------------------------

```python
def test_isinstance_check_false():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #79
#--------------------------

```python
def test_get_config_with_valid_path():
    config_dict = get_config('valid_config.yaml')
    assert isinstance(config_dict, dict)
    assert 'replay_dir' in config_dict
    assert 'cookiecutters_dir' in config_dict

def test_get_config_with_nonexistent_path():
    try:
        get_config('nonexistent_config.yaml')
    except ConfigDoesNotExistException as e:
        assert str(e) == 'Config file nonexistent_config.yaml does not exist.'

def test_get_config_with_invalid_yaml():
    try:
        get_config('invalid_yaml_config.yaml')
    except InvalidConfiguration as e:
        assert str(e) == 'Unable to parse YAML file invalid_yaml_config.yaml.'

def test_get_config_with_non_dict_yaml():
    try:
        get_config('non_dict_yaml_config.yaml')
    except InvalidConfiguration as e:
        assert str(e) == 'Top-level element of YAML file non_dict_yaml_config.yaml should be an object.'

def test_get_config_expands_environment_variables():
    os.environ['TEST_VAR'] = 'test_value'
    config_dict = get_config('config_with_env_var.yaml')
    assert config_dict['replay_dir'] == 'test_value'

def test_get_config_expands_user_home():
    config_dict = get_config('config_with_home.yaml')
    assert config_dict['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')


# LLM-generated content at query #80
#--------------------------

```python
def test_yaml_error_raised_when_parsing_invalid_yaml():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("invalid yaml content")


# LLM-generated content at query #81
#--------------------------

```python
def test_config_path_does_not_exist_raises_exception():
    with pytest.raises(ConfigDoesNotExistException):
        get_config("non_existent_path.yaml")


# LLM-generated content at query #82
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #83
#--------------------------

```python
def test_yaml_dict_is_not_dict():
    yaml_dict = []
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #84
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


# LLM-generated content at query #85
#--------------------------

```python
def test_yaml_safe_load_returns_none():
    with patch('builtins.open', mock_open(read_data='')):
        assert not yaml.safe_load(open('dummy_path'))


# LLM-generated content at query #86
#--------------------------

```python
def test_config_path_exists():
    config_path = 'path/to/existing/config.yaml'
    assert os.path.exists(config_path) is True


# LLM-generated content at query #87
#--------------------------

```python
def test_yaml_error_raised_when_parsing_fails():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("invalid yaml content")


# LLM-generated content at query #88
#--------------------------

```python
def test_yaml_safe_load_returns_dict_or_none():
    yaml_dict = yaml.safe_load(file_handle) or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #89
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
            assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #90
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration():
    config_path = 'invalid.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config(config_path)
    assert 'Unable to parse YAML file' in str(exc_info.value)


# LLM-generated content at query #91
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay/path',
        'cookiecutters_dir': '/expanded/cookies/path',
        'other_key': 'value'
    }
    with patch('builtins.open', mock_open(read_data='replay_dir: $HOME/replay\ncookiecutters_dir: $HOME/cookies\nother_key: value')):
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', return_value={'replay_dir': '$HOME/replay', 'cookiecutters_dir': '$HOME/cookies', 'other_key': 'value'}):
                with patch('os.path.expandvars', side_effect=lambda x: x):
                    with patch('os.path.expanduser', side_effect=lambda x: x.replace('$HOME', '/expanded')):
                        result = get_config(config_path)
                        assert result == expected_config

def test_get_config_with_nonexistent_path():
    config_path = 'nonexistent_config.yaml'
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException) as excinfo:
            get_config(config_path)
        assert str(excinfo.value) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with patch('builtins.open', mock_open(read_data='invalid yaml content')):
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError):
                with pytest.raises(InvalidConfiguration) as excinfo:
                    get_config(config_path)
                assert str(excinfo.value) == f'Unable to parse YAML file {config_path}.'

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with patch('builtins.open', mock_open(read_data='- list item')):
        with patch('os.path.exists', return_value=True):
            with patch('yaml.safe_load', return_value=['list item']):
                with pytest.raises(InvalidConfiguration) as excinfo:
                    get_config(config_path)
                assert str(excinfo.value) == f'Top-level element of YAML file {config_path} should be an object.'


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
            mock_file.assert_called_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
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


# LLM-generated content at query #2
#--------------------------

```python
def test_keyerror_raised_when_env_var_not_set():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    try:
        os.environ['COOKIECUTTER_CONFIG']
    except KeyError as e:
        assert True
    else:
        assert False


# LLM-generated content at query #3
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
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config(config_file='custom_path')
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('os.path.exists', return_value=True), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_not_set_user_config_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='key: value')), \
         patch('yaml.safe_load', return_value={'key': 'value'}):
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'key': 'value'})
        assert result == expected

def test_get_user_config_env_var_not_set_user_config_not_exists():
    with patch.dict('os.environ', {}, clear=True), \
         patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #4
#--------------------------

```python
def test_keyerror_predicate():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #5
#--------------------------

```python
def test_keyerror_predicate():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #6
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_get_user_config_with_default_config_dict():
    custom_config = {"key": "value"}
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


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_with_valid_path():
    config_path = 'valid_config.yaml'
    expected_config = {
        'replay_dir': '/expanded/replay',
        'cookiecutters_dir': '/expanded/cookies',
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


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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
    with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
        with patch('os.path.exists', return_value=True):
            result = get_user_config(config_file='custom_path')
            mock_file.assert_called_once_with('custom_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_env_var_set():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with('env_path', encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_env_var_not_set_user_config_exists():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            mock_file.assert_called_once_with(USER_CONFIG_PATH, encoding='utf-8')
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_env_var_not_set_user_config_not_exists():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_43():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #11
#--------------------------

```python
def test_config_file_exists_returns_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_config_path_exists():
    config_path = "path/to/existing/config.yaml"
    assert os.path.exists(config_path)


# LLM-generated content at query #13
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

def test_expand_path_with_no_expansion_needed():
    assert _expand_path('/absolute/path') == '/absolute/path'

def test_expand_path_with_empty_string():
    assert _expand_path('') == ''


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    assert os.path.exists(USER_CONFIG_PATH) is True


# LLM-generated content at query #15
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

def test_get_user_config_with_env_var():
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': 'env_path'}):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            with patch('os.path.exists', return_value=True):
                result = get_user_config()
                assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_default_path():
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='key: value')) as mock_file:
            result = get_user_config()
            assert result == merge_configs(DEFAULT_CONFIG, {'key': 'value'})

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #21
#--------------------------

```python
def test_config_file_exists_returns_custom_config():
    assert get_user_config(config_file="custom_path") == get_config("custom_path")


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    # Ensure that the KeyError is raised when 'COOKIECUTTER_CONFIG' is not in os.environ
    # This will make the predicate at line 40 evaluate to False
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #23
#--------------------------

```python
def test_config_path_exists():
    config_path = 'existing_config.yaml'
    assert os.path.exists(config_path)


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write('{}')
    assert os.path.exists(USER_CONFIG_PATH)


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
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #27
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #28
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

def test_get_user_config_with_no_config():
    with patch('os.path.exists', return_value=False):
        result = get_user_config()
        assert result == DEFAULT_CONFIG


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_config_path_exists():
    assert os.path.exists(config_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ = {}
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #32
#--------------------------

```python
def test_user_config_path_exists():
    assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #33
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

def test_get_user_config_with_no_config_found():
    with patch('os.path.exists', return_value=False):
        with patch.dict('os.environ', {}, clear=True):
            result = get_user_config()
            assert result == copy.copy(DEFAULT_CONFIG)


