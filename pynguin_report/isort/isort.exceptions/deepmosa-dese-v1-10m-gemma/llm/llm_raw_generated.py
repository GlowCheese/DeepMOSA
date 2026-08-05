####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_initialization():
    unsupported_settings = {
        "line_length": {"value": 88, "source": "pyproject.toml"},
        "multi_line_output": {"value": 3, "source": "cli"}
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(error)
    assert "\t- line_length = 88  (source: 'pyproject.toml')" in str(error)
    assert "\t- multi_line_output = 3  (source: 'cli')" in str(error)

def test_unsupported_settings_constructor_empty_dict():
    unsupported_settings = {}
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:\n\n\n" in str(error)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_formats_message_correctly():
    unsupported_settings = {
        "line_length": {"value": 88, "source": "pyproject.toml"},
        "multi_line_output": {"value": 3, "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    
    expected_message_part1 = "isort was provided settings that it doesn't support:\n\n"
    expected_option1 = "\t- line_length = 88  (source: 'pyproject.toml')"
    expected_option2 = "\t- multi_line_output = 3  (source: 'cli')"
    expected_message_part2 = "\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    
    assert exception.unsupported_settings == unsupported_settings
    assert expected_message_part1 + expected_option1 + "\n" + expected_option2 + expected_message_part2 in str(exception)

def test_unsupported_settings_constructor_empty_dict():
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


