####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_unsupported_settings_constructor():
    unsupported_settings = {"setting1": {"value": "value1", "source": "config"}, "setting2": {"value": "value2", "source": "cli"}}
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    expected_message = "isort was provided settings that it doesn't support:\n\n\t- setting1 = value1  (source: 'config')\n\t- setting2 = value2  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    assert str(exception) == expected_message


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_unsupported_settings_constructor():
    unsupported_settings = {"setting1": {"value": "value1", "source": "config"}, "setting2": {"value": "value2", "source": "cli"}}
    exception = UnsupportedSettings(unsupported_settings)
    expected_message = "isort was provided settings that it doesn't support:\n\n\t- setting1 = value1  (source: 'config')\n\t- setting2 = value2  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    assert str(exception) == expected_message
    assert exception.unsupported_settings == unsupported_settings


