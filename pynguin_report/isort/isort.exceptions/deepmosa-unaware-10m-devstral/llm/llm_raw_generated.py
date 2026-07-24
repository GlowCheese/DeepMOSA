####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ProfileDoesNotExist():
    profile = "nonexistent_profile"
    exception = ProfileDoesNotExist(profile)
    assert exception.profile == profile
    assert str(exception) == (
        f"Specified profile of {profile} does not exist. "
        f"Available profiles: {','.join(profiles)}."
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_FormattingPluginDoesNotExist():
    formatter = "nonexistent_formatter"
    exception = FormattingPluginDoesNotExist(formatter)
    assert str(exception) == f"Specified formatting plugin of {formatter} does not exist. "
    assert exception.formatter == formatter


# LLM-generated content at query #3
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )
    assert error.file_path == file_path


# LLM-generated content at query #4
#--------------------------

```python
def test_LiteralParsingFailure():
    code = "invalid_literal"
    original_error = ValueError("Invalid literal")
    exception = LiteralParsingFailure(code, original_error)

    assert str(exception) == (
        f"isort failed to parse the given literal {code}. It's important to note "
        "that isort literal sorting only supports simple literals parsable by "
        f"ast.literal_eval which gave the exception of {original_error}."
    )
    assert exception.code == code
    assert exception.original_error == original_error


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ProfileDoesNotExist():
    profile = "non_existent_profile"
    exception = ProfileDoesNotExist(profile)
    assert exception.profile == profile
    assert str(exception) == (
        f"Specified profile of {profile} does not exist. "
        f"Available profiles: {','.join(profiles)}."
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_FormattingPluginDoesNotExist():
    formatter = "nonexistent_formatter"
    exception = FormattingPluginDoesNotExist(formatter)
    assert str(exception) == f"Specified formatting plugin of {formatter} does not exist. "
    assert exception.formatter == formatter


# LLM-generated content at query #3
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert error.file_path == file_path
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )


# LLM-generated content at query #4
#--------------------------

```python
def test_LiteralParsingFailure():
    code = "invalid_literal"
    original_error = ValueError("Invalid literal")
    exception = LiteralParsingFailure(code, original_error)
    assert exception.code == code
    assert exception.original_error == original_error
    assert str(exception) == (
        f"isort failed to parse the given literal {code}. It's important to note "
        "that isort literal sorting only supports simple literals parsable by "
        f"ast.literal_eval which gave the exception of {original_error}."
    )


# LLM-generated content at query #5
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert error.file_path == file_path
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )


# LLM-generated content at query #6
#--------------------------

```python
def test_FileSkipComment():
    file_path = "test_file.py"
    skip_comment = FileSkipComment(file_path)
    assert skip_comment.file_path == file_path
    assert str(skip_comment) == f"{file_path} contains a file skip comment and was skipped."


# LLM-generated content at query #7
#--------------------------

```python
def test_ISortError___reduce__():
    error = ISortError("Test error message")
    error.custom_attr = "custom_value"
    reconstructed_error = error.__reduce__()[0]()

    assert isinstance(reconstructed_error, ISortError)
    assert str(reconstructed_error) == "Test error message"
    assert reconstructed_error.custom_attr == "custom_value"


# LLM-generated content at query #8
#--------------------------

```python
def test_MissingSection():
    import_module = "test_module"
    section = "test_section"
    error = MissingSection(import_module, section)
    assert error.import_module == import_module
    assert error.section == section
    assert str(error) == (
        f"Found {import_module} import while parsing, but {section} was not included "
        "in the `sections` setting of your config. Please add it before continuing\n"
        "See https://pycqa.github.io/isort/#custom-sections-and-ordering "
        "for more info."
    )


# LLM-generated content at query #9
#--------------------------

```python
def test_MissingSection():
    import_module = "test_module"
    section = "test_section"
    error = MissingSection(import_module, section)
    assert error.import_module == import_module
    assert error.section == section
    assert str(error) == (
        f"Found {import_module} import while parsing, but {section} was not included "
        "in the `sections` setting of your config. Please add it before continuing\n"
        "See https://pycqa.github.io/isort/#custom-sections-and-ordering "
        "for more info."
    )


# LLM-generated content at query #10
#--------------------------

```python
def test_FileSkipSetting():
    file_path = "test_file.py"
    file_skip_setting = FileSkipSetting(file_path=file_path)
    assert file_skip_setting.file_path == file_path
    assert str(file_skip_setting) == (
        f"{file_path} was skipped as it's listed in 'skip' setting"
        " or matches a glob in 'skip_glob' setting"
    )


# LLM-generated content at query #11
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )
    assert error.file_path == file_path


# LLM-generated content at query #12
#--------------------------

```python
def test_UnsupportedSettings():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "config"},
        "setting2": {"value": "value2", "source": "CLI"},
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert str(error) == (
        "isort was provided settings that it doesn't support:\n\n"
        "\t- setting1 = value1  (source: 'config')\n"
        "\t- setting2 = value2  (source: 'CLI')\n\n"
        "For a complete and up-to-date listing of supported settings see: "
        "https://pycqa.github.io/isort/docs/configuration/options.\n"
    )


# LLM-generated content at query #13
#--------------------------

```python
def test_FileSkipSetting():
    file_path = "test_file.py"
    skip_setting = FileSkipSetting(file_path=file_path)
    assert skip_setting.file_path == file_path
    assert str(skip_setting) == (
        f"{file_path} was skipped as it's listed in 'skip' setting"
        " or matches a glob in 'skip_glob' setting"
    )


# LLM-generated content at query #14
#--------------------------

```python
def test_InvalidSettingsPath():
    settings_path = "/non/existent/path"
    exception = InvalidSettingsPath(settings_path)
    assert exception.settings_path == settings_path
    assert str(exception) == (
        f"isort was told to use the settings_path: {settings_path} as the base directory or "
        "file that represents the starting point of config file discovery, but it does not "
        "exist."
    )
    assert isinstance(exception, ISortError)


# LLM-generated content at query #15
#--------------------------

```python
def test_LiteralParsingFailure():
    code = "invalid_literal"
    original_error = ValueError("invalid literal for int() with base 10: 'invalid_literal'")
    exception = LiteralParsingFailure(code, original_error)
    assert str(exception) == (
        f"isort failed to parse the given literal {code}. It's important to note "
        "that isort literal sorting only supports simple literals parsable by "
        f"ast.literal_eval which gave the exception of {original_error}."
    )
    assert exception.code == code
    assert exception.original_error == original_error


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert error.file_path == file_path
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_SortingFunctionDoesNotExist():
    sort_order = "invalid_sort"
    available_sort_orders = ["default", "length", "alphabetical"]
    exception = SortingFunctionDoesNotExist(sort_order, available_sort_orders)
    assert exception.sort_order == sort_order
    assert exception.available_sort_orders == available_sort_orders
    assert str(exception) == (
        f"Specified sort_order of {sort_order} does not exist. "
        f"Available sort_orders: {','.join(available_sort_orders)}."
    )


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidSettingsPath():
    settings_path = "/non/existent/path"
    exception = InvalidSettingsPath(settings_path)
    assert exception.settings_path == settings_path
    assert str(exception) == (
        "isort was told to use the settings_path: /non/existent/path as the base directory or "
        "file that represents the starting point of config file discovery, but it does not exist."
    )


# LLM-generated content at query #4
#--------------------------

```python
def test_IntroducedSyntaxErrors():
    file_path = "test_file.py"
    error = IntroducedSyntaxErrors(file_path)
    assert error.file_path == file_path
    assert str(error) == (
        f"isort introduced syntax errors when attempting to sort the imports contained within "
        f"{file_path}."
    )


# LLM-generated content at query #5
#--------------------------

```python
def test_LiteralSortTypeMismatch():
    kind = str
    expected_kind = int
    error = LiteralSortTypeMismatch(kind, expected_kind)
    assert error.kind == kind
    assert error.expected_kind == expected_kind
    assert str(error) == (
        f"isort was told to sort a literal of type {expected_kind} but was given "
        f"a literal of type {kind}."
    )


# LLM-generated content at query #6
#--------------------------

```python
def test_ProfileDoesNotExist():
    profile = "nonexistent_profile"
    exception = ProfileDoesNotExist(profile)
    assert str(exception) == (
        f"Specified profile of {profile} does not exist. "
        f"Available profiles: {','.join(profiles)}."
    )
    assert exception.profile == profile


# LLM-generated content at query #7
#--------------------------

```python
def test_FileSkipSetting():
    file_path = "test_file.py"
    skip_setting = FileSkipSetting(file_path=file_path)
    assert skip_setting.file_path == file_path
    assert str(skip_setting) == (
        f"{file_path} was skipped as it's listed in 'skip' setting"
        " or matches a glob in 'skip_glob' setting"
    )
    assert isinstance(skip_setting, FileSkipped)
    assert isinstance(skip_setting, ISortError)


# LLM-generated content at query #8
#--------------------------

```python
def test_AssignmentsFormatMismatch():
    code = "x = 1\ny = 2\nz = 3"
    exception = AssignmentsFormatMismatch(code)
    assert str(exception) == (
        "isort was told to sort a section of assignments, however the given code:\n\n"
        f"{code}\n\n"
        "Does not match isort's strict single line formatting requirement for assignment "
        "sorting:\n\n"
        "{variable_name} = {value}\n"
        "{variable_name2} = {value2}\n"
        "...\n\n"
    )
    assert exception.code == code


# LLM-generated content at query #9
#--------------------------

```python
def test_MissingSection():
    import_module = "test_module"
    section = "test_section"
    error = MissingSection(import_module, section)
    assert error.import_module == import_module
    assert error.section == section
    assert str(error) == (
        f"Found {import_module} import while parsing, but {section} was not included "
        "in the `sections` setting of your config. Please add it before continuing\n"
        "See https://pycqa.github.io/isort/#custom-sections-and-ordering "
        "for more info."
    )


# LLM-generated content at query #10
#--------------------------

```python
def test_ISortError():
    error = ISortError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)
    assert error.__reduce__() == (partial(type(error), **error.__dict__), ())


# LLM-generated content at query #11
#--------------------------

```python
def test_ExistingSyntaxErrors():
    file_path = "test_file.py"
    error = ExistingSyntaxErrors(file_path)
    assert str(error) == (
        f"isort was told to sort imports within code that contains syntax errors: {file_path}."
    )
    assert error.file_path == file_path


# LLM-generated content at query #12
#--------------------------

```python
def test_FileSkipComment():
    file_path = "test_file.py"
    exception = FileSkipComment(file_path)
    assert exception.file_path == file_path
    assert str(exception) == f"{file_path} contains a file skip comment and was skipped."
    assert isinstance(exception, FileSkipped)
    assert isinstance(exception, ISortError)


# LLM-generated content at query #13
#--------------------------

```python
def test_FormattingPluginDoesNotExist():
    formatter = "nonexistent_formatter"
    exception = FormattingPluginDoesNotExist(formatter)
    assert str(exception) == f"Specified formatting plugin of {formatter} does not exist. "
    assert exception.formatter == formatter


# LLM-generated content at query #14
#--------------------------

```python
def test_LiteralSortTypeMismatch():
    kind = int
    expected_kind = str
    exception = LiteralSortTypeMismatch(kind, expected_kind)

    assert exception.kind == kind
    assert exception.expected_kind == expected_kind
    assert str(exception) == (
        f"isort was told to sort a literal of type {expected_kind} but was given "
        f"a literal of type {kind}."
    )


# LLM-generated content at query #15
#--------------------------

```python
def test_ISortError():
    error = ISortError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)
    assert error.__reduce__() == (partial(type(error), **error.__dict__), ())


# LLM-generated content at query #16
#--------------------------

```python
def test_UnsupportedEncoding():
    filename = "test_file.py"
    error = UnsupportedEncoding(filename)
    assert str(error) == f"Unknown or unsupported encoding in {filename}"
    assert error.filename == filename


# LLM-generated content at query #17
#--------------------------

```python
def test_SortingFunctionDoesNotExist():
    sort_order = "invalid_sort_order"
    available_sort_orders = ["order1", "order2", "order3"]

    exception = SortingFunctionDoesNotExist(sort_order, available_sort_orders)

    assert exception.sort_order == sort_order
    assert exception.available_sort_orders == available_sort_orders
    assert str(exception) == (
        f"Specified sort_order of {sort_order} does not exist. "
        f"Available sort_orders: {','.join(available_sort_orders)}."
    )


# LLM-generated content at query #18
#--------------------------

```python
def test_SortingFunctionDoesNotExist():
    sort_order = "invalid_sort"
    available_sort_orders = ["default", "length", "alphabetical"]
    exception = SortingFunctionDoesNotExist(sort_order, available_sort_orders)
    assert exception.sort_order == sort_order
    assert exception.available_sort_orders == available_sort_orders
    assert str(exception) == (
        f"Specified sort_order of {sort_order} does not exist. "
        f"Available sort_orders: {','.join(available_sort_orders)}."
    )


# LLM-generated content at query #19
#--------------------------

```python
def test_ProfileDoesNotExist():
    profile = "nonexistent_profile"
    exception = ProfileDoesNotExist(profile)
    assert exception.profile == profile
    assert str(exception) == (
        f"Specified profile of {profile} does not exist. "
        f"Available profiles: {','.join(profiles)}."
    )


# LLM-generated content at query #20
#--------------------------

```python
def test_FileSkipped():
    message = "Test skip message"
    file_path = "test_file.py"

    exception = FileSkipped(message, file_path)

    assert str(exception) == message
    assert exception.message == message
    assert exception.file_path == file_path


# LLM-generated content at query #21
#--------------------------

```python
def test_ISortError():
    error = ISortError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)
    assert isinstance(error, ISortError)


# LLM-generated content at query #22
#--------------------------

```python
def test_ISortError___reduce__():
    error = ISortError("test message")
    error.custom_attr = "custom_value"
    reconstructed_error = error.__reduce__()[0]()
    assert isinstance(reconstructed_error, ISortError)
    assert str(reconstructed_error) == "test message"
    assert reconstructed_error.custom_attr == "custom_value"


# LLM-generated content at query #23
#--------------------------

```python
def test_UnsupportedSettings():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "config"},
        "setting2": {"value": 42, "source": "CLI"},
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert "setting1 = value1" in str(error)
    assert "setting2 = 42" in str(error)
    assert "source: 'config'" in str(error)
    assert "source: 'CLI'" in str(error)


# LLM-generated content at query #24
#--------------------------

```python
def test_FileSkipSetting():
    file_path = "test_file.py"
    skip_setting = FileSkipSetting(file_path)
    assert skip_setting.file_path == file_path
    assert str(skip_setting) == (
        f"{file_path} was skipped as it's listed in 'skip' setting"
        " or matches a glob in 'skip_glob' setting"
    )


# LLM-generated content at query #25
#--------------------------

```python
def test_ISortError___reduce__():
    # Test basic ISortError reduction
    error = ISortError("Test error message")
    reduced = error.__reduce__()
    assert reduced[0] == partial(type(error), **error.__dict__)
    assert reduced[1] == ()

    # Test reduction preserves attributes
    error_with_attrs = ISortError("Another message")
    error_with_attrs.custom_attr = "custom_value"
    reduced_with_attrs = error_with_attrs.__reduce__()
    reconstructed = reduced_with_attrs[0](*reduced_with_attrs[1])
    assert reconstructed.custom_attr == "custom_value"
    assert str(reconstructed) == "Another message"

    # Test reduction of subclass
    subclass_error = InvalidSettingsPath("/nonexistent/path")
    reduced_subclass = subclass_error.__reduce__()
    assert reduced_subclass[0] == partial(type(subclass_error), **subclass_error.__dict__)
    reconstructed_subclass = reduced_subclass[0](*reduced_subclass[1])
    assert isinstance(reconstructed_subclass, InvalidSettingsPath)
    assert reconstructed_subclass.settings_path == "/nonexistent/path"


