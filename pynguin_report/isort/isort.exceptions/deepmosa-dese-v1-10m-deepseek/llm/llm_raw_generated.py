####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_UnsupportedSettings_constructor():
    unsupported_settings = {
        "option1": {"value": "value1", "source": "config"},
        "option2": {"value": "value2", "source": "CLI"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    assert str(exception) == (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- option1 = value1  (source: 'config')\n"
        "\t- option2 = value2  (source: 'CLI')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unsupported_settings_init():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "config"},
        "setting2": {"value": "value2", "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support" in str(exception)
    assert "setting1 = value1  (source: 'config')" in str(exception)
    assert "setting2 = value2  (source: 'cli')" in str(exception)
    assert "https://pycqa.github.io/isort/docs/configuration/options" in str(exception)


