####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "unknown_option": {"value": "some_value", "source": "setup.cfg"},
        "invalid_setting": {"value": 42, "source": "command_line"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)
    assert "unknown_option = some_value  (source: 'setup.cfg')" in str(exception)
    assert "invalid_setting = 42  (source: 'command_line')" in str(exception)
    assert "https://pycqa.github.io/isort/docs/configuration/options" in str(exception)


def test_unsupported_settings_constructor_empty():
    unsupported_settings = {}
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:" in str(exception)


def test_unsupported_settings_constructor_single_setting():
    unsupported_settings = {
        "bad_option": {"value": "test_value", "source": ".isort.cfg"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "bad_option = test_value  (source: '.isort.cfg')" in str(exception)
    assert len(exception.unsupported_settings) == 1


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "invalid_option": {"value": "test_value", "source": "config.ini"},
        "another_invalid": {"value": 42, "source": "CLI"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)
    assert "invalid_option = test_value  (source: 'config.ini')" in str(exception)
    assert "another_invalid = 42  (source: 'CLI')" in str(exception)
    assert "https://pycqa.github.io/isort/docs/configuration/options" in str(exception)


def test_unsupported_settings_constructor_empty():
    unsupported_settings = {}
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:" in str(exception)


def test_unsupported_settings_constructor_single_setting():
    unsupported_settings = {
        "bad_setting": {"value": "some_value", "source": "runtime"}
    }
    
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == unsupported_settings
    assert "bad_setting = some_value  (source: 'runtime')" in str(exception)


