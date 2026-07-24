####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FileSkipped():
    message = "Test skip message"
    file_path = "test_file.py"
    exception = FileSkipped(message, file_path)
    assert str(exception) == message
    assert exception.message == message
    assert exception.file_path == file_path


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_InvalidSettingsPath():
    settings_path = "invalid/path"
    exception = InvalidSettingsPath(settings_path)
    assert exception.settings_path == settings_path
    assert str(exception) == (
        "isort was told to use the settings_path: invalid/path as the base directory or "
        "file that represents the starting point of config file discovery, but it does not "
        "exist."
    )


# LLM-generated content at query #4
#--------------------------

```python
def test_InvalidSettingsPath():
    settings_path = "non_existent_path"
    exception = InvalidSettingsPath(settings_path)
    assert exception.settings_path == settings_path
    assert str(exception) == (
        "isort was told to use the settings_path: non_existent_path as the base directory or "
        "file that represents the starting point of config file discovery, but it does not "
        "exist."
    )


# LLM-generated content at query #5
#--------------------------

```python
def test_SortingFunctionDoesNotExist():
    sort_order = "invalid_sort"
    available_sort_orders = ["order1", "order2", "order3"]
    exception = SortingFunctionDoesNotExist(sort_order, available_sort_orders)
    assert exception.sort_order == sort_order
    assert exception.available_sort_orders == available_sort_orders
    assert str(exception) == (
        f"Specified sort_order of {sort_order} does not exist. "
        f"Available sort_orders: {','.join(available_sort_orders)}."
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
def test_UnsupportedEncoding():
    filename = "test_file.py"
    error = UnsupportedEncoding(filename)
    assert str(error) == f"Unknown or unsupported encoding in {filename}"
    assert error.filename == filename


# LLM-generated content at query #8
#--------------------------

```python
def test_InvalidSettingsPath():
    settings_path = "non_existent_path"
    exception = InvalidSettingsPath(settings_path)
    assert exception.settings_path == settings_path
    assert str(exception) == (
        "isort was told to use the settings_path: non_existent_path as the base directory or "
        "file that represents the starting point of config file discovery, but it does not exist."
    )
    assert isinstance(exception, ISortError)


# LLM-generated content at query #9
#--------------------------

```python
def test_FileSkipComment():
    file_path = "test_file.py"
    file_skip_comment = FileSkipComment(file_path)
    assert file_skip_comment.file_path == file_path
    assert str(file_skip_comment) == f"{file_path} contains a file skip comment and was skipped."
    assert isinstance(file_skip_comment, FileSkipped)
    assert isinstance(file_skip_comment, ISortError)


# LLM-generated content at query #10
#--------------------------

```python
def test_UnsupportedSettings():
    unsupported_settings = {
        "setting1": {"value": "value1", "source": "config"},
        "setting2": {"value": 42, "source": "CLI"},
    }
    error = UnsupportedSettings(unsupported_settings)
    assert error.unsupported_settings == unsupported_settings
    assert "isort was provided settings that it doesn't support:" in str(error)
    assert "setting1 = value1  (source: 'config')" in str(error)
    assert "setting2 = 42  (source: 'CLI')" in str(error)
    assert "https://pycqa.github.io/isort/docs/configuration/options." in str(error)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_FormattingPluginDoesNotExist():
    formatter = "nonexistent_formatter"
    exception = FormattingPluginDoesNotExist(formatter)
    assert str(exception) == f"Specified formatting plugin of {formatter} does not exist. "
    assert exception.formatter == formatter


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
def test_ISortError___reduce__():
    # Test basic ISortError
    error = ISortError("Test message")
    reconstructed_error = error.__reduce__()[0]()
    assert isinstance(reconstructed_error, ISortError)
    assert str(reconstructed_error) == "Test message"

    # Test ISortError with custom attributes
    error = ISortError("Test message")
    error.custom_attr = "custom_value"
    reconstructed_error = error.__reduce__()[0]()
    assert isinstance(reconstructed_error, ISortError)
    assert str(reconstructed_error) == "Test message"
    assert reconstructed_error.custom_attr == "custom_value"

    # Test ISortError subclass
    error = InvalidSettingsPath("invalid/path")
    reconstructed_error = error.__reduce__()[0]()
    assert isinstance(reconstructed_error, InvalidSettingsPath)
    assert str(reconstructed_error).startswith("isort was told to use the settings_path:")
    assert reconstructed_error.settings_path == "invalid/path"


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
def test_SortingFunctionDoesNotExist():
    sort_order = "invalid_sort"
    available_sort_orders = ["order1", "order2", "order3"]
    exception = SortingFunctionDoesNotExist(sort_order, available_sort_orders)
    assert exception.sort_order == sort_order
    assert exception.available_sort_orders == available_sort_orders
    assert str(exception) == (
        f"Specified sort_order of {sort_order} does not exist. "
        f"Available sort_orders: {','.join(available_sort_orders)}."
    )


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_UnsupportedEncoding():
    filename = "test_file.py"
    error = UnsupportedEncoding(filename)
    assert error.filename == filename
    assert str(error) == f"Unknown or unsupported encoding in {filename}"


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
    assert "isort was provided settings that it doesn't support:" in str(error)
    assert "setting1 = value1  (source: 'config')" in str(error)
    assert "setting2 = value2  (source: 'CLI')" in str(error)
    assert "https://pycqa.github.io/isort/docs/configuration/options." in str(error)


# LLM-generated content at query #13
#--------------------------

```python
def test_AssignmentsFormatMismatch():
    code = "invalid_code"
    exception = AssignmentsFormatMismatch(code)
    assert exception.code == code
    assert str(exception) == (
        "isort was told to sort a section of assignments, however the given code:\n\n"
        f"{code}\n\n"
        "Does not match isort's strict single line formatting requirement for assignment "
        "sorting:\n\n"
        "{variable_name} = {value}\n"
        "{variable_name2} = {value2}\n"
        "...\n\n"
    )


# LLM-generated content at query #14
#--------------------------

```python
def test_FormattingPluginDoesNotExist():
    formatter = "nonexistent_formatter"
    exception = FormattingPluginDoesNotExist(formatter)
    assert str(exception) == f"Specified formatting plugin of {formatter} does not exist. "
    assert exception.formatter == formatter


# LLM-generated content at query #15
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


