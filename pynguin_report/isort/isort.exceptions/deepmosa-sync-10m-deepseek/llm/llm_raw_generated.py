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

def test_unsupported_settings_constructor_with_single_setting():
    unsupported_settings = {"setting1": {"value": "value1", "source": "config"}}
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)
    assert "setting1 = value1  (source: 'config')" in str(exception)

def test_unsupported_settings_constructor_with_multiple_settings():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "config"},
        "setting2": {"value": "value2", "source": "cli"}
    }
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    exception_str = str(exception)
    assert "isort was provided settings that it doesn't support:" in exception_str
    assert "setting1 = value1  (source: 'config')" in exception_str
    assert "setting2 = value2  (source: 'cli')" in exception_str

def test_unsupported_settings_constructor_with_empty_dict():
    unsupported_settings = {}
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(exception)

def test_unsupported_settings_constructor_check_exception_inheritance():
    unsupported_settings = {"setting1": {"value": "value1", "source": "config"}}
    exception = UnsupportedSettings(unsupported_settings)
    assert isinstance(exception, ISortError)

def test_unsupported_settings_constructor_with_special_characters():
    unsupported_settings = {"setting@name": {"value": "value@123", "source": "file.json"}}
    exception = UnsupportedSettings(unsupported_settings)
    assert exception.unsupported_settings == unsupported_settings
    assert "setting@name = value@123  (source: 'file.json')" in str(exception)


