####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "unknown_option": {"value": "test_value", "source": "config.ini"},
        "invalid_setting": {"value": 42, "source": "CLI"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)
    assert "unknown_option = test_value  (source: 'config.ini')" in str(exception)
    assert "invalid_setting = 42  (source: 'CLI')" in str(exception)
    assert "https://pycqa.github.io/isort/docs/configuration/options" in str(exception)


def test_unsupported_settings_constructor_empty():
    unsupported_settings = {}
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)


def test_unsupported_settings_constructor_single_setting():
    unsupported_settings = {
        "bad_option": {"value": "some_value", "source": "environment"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "bad_option = some_value  (source: 'environment')" in str(exception)
    assert isinstance(exception, Exception)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "setting1": {"value": "val1", "source": "config.ini"},
        "setting2": {"value": "val2", "source": "command_line"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)
    assert "setting1 = val1  (source: 'config.ini')" in str(exception)
    assert "setting2 = val2  (source: 'command_line')" in str(exception)
    assert "https://pycqa.github.io/isort/docs/configuration/options" in str(exception)


def test_unsupported_settings_constructor_empty():
    unsupported_settings = {}
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:" in str(exception)


def test_unsupported_settings_constructor_single_setting():
    unsupported_settings = {
        "unknown_option": {"value": "some_value", "source": "pyproject.toml"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "unknown_option = some_value  (source: 'pyproject.toml')" in str(exception)


