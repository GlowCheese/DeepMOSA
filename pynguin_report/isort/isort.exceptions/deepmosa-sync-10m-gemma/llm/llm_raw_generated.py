####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_formats_error_message_correctly():
    unsupported_settings = {
        "profile": {"value": "black", "source": "config"},
        "line_length": {"value": 88, "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    
    expected_message = (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- profile = black  (source: 'config')\n"
        "\t- line_length = 88  (source: 'cli')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )
    
    assert exception.unsupported_settings == unsupported_settings
    assert str(exception) == expected_message

def test_unsupported_settings_constructor_with_empty_dict():
    unsupported_settings = {}
    exception = UnsupportedSettings(unsupported_settings)
    
    expected_message = (
        "isort was provided settings that it doesn't support:\n\n"
        "\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )
    
    assert exception.unsupported_settings == {}
    assert str(exception) == expected_message
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_message_formatting():
    unsupported_settings = {
        "some_option": {"value": "some_value", "source": "config"},
        "another_option": {"value": True, "source": "cli"}
    }
    error = UnsupportedSettings(unsupported_settings)
    expected_substring1 = "\t- some_option = some_value  (source: 'config')"
    expected_substring2 = "\t- another_option = True  (source: 'cli')"
    assert error.unsupported_settings == unsupported_settings
    assert expected_substring1 in str(error)
    assert expected_substring2 in str(error)
    assert "isort was provided settings that it doesn't support:" in str(error)

def test_unsupported_settings_constructor_empty_dict():
    unsupported_settings = {}
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:\n\n\n" in str(error)
```


