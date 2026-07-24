####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_formats_error_message_correctly():
    unsupported_settings = {
        "line_length": {"value": "88", "source": "pyproject.toml"},
        "multi_line_output": {"value": "3", "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    
    expected_error_part1 = "isort was provided settings that it doesn't support:\n\n"
    expected_line1 = "\t- line_length = 88  (source: 'pyproject.toml')"
    expected_line2 = "\t- multi_line_output = 3  (source: 'cli')"
    expected_error_part2 = "\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    
    assert exception.unsupported_settings == unsupported_settings
    assert expected_error_part1 + expected_line1 + "\n" + expected_line2 + expected_error_part2 in str(exception)

def test_unsupported_settings_constructor_handles_empty_dict():
    unsupported_settings = {}
    exception = UnsupportedSettings(unsupported_settings)
    
    assert exception.unsupported_settings == {}
    assert "isort was provided settings that it doesn't support:\n\n\n\nFor a complete" in str(exception)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor_message_format():
    unsupported_settings = {
        "line_length": {"value": "88", "source": "config"},
        "multi_line_output": {"value": "3", "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    expected_message = (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- line_length = 88  (source: 'config')\n"
        "\t- multi_line_output = 3  (source: 'cli')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )
    assert str(exception) == expected_message
    assert exception.unsupported_settings == unsupported_settings

def test_unsupported_settings_constructor_empty_dict():
    unsupported_settings = {}
    exception = UnsupportedSettings(unsupported_settings)
    expected_message = (
        "isort was provided settings that it doesn't support:\n\n"
        "\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )
    assert str(exception) == expected_message
    assert exception.unsupported_settings == {}
```


