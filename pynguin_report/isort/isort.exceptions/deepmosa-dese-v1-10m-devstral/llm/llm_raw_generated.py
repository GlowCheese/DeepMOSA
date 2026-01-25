####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "source1"},
        "setting2": {"value": "value2", "source": "source2"}
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert str(error) == (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- setting1 = value1  (source: 'source1')\n"
        "\t- setting2 = value2  (source: 'source2')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_constructor():
    unsupported_settings = {
        "setting1": {"value": "invalid_value", "source": "config"},
        "setting2": {"value": 42, "source": "CLI"}
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert str(error) == (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- setting1 = invalid_value  (source: 'config')\n"
        "\t- setting2 = 42  (source: 'CLI')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )


