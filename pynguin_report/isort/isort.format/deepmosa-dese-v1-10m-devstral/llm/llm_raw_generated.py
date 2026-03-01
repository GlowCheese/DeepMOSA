####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_create_terminal_printer_with_color_and_colorama_available():
    color = True
    output = None
    error = "Error: {error} - {message}"
    success = "Success: {success} - {message}"
    result = create_terminal_printer(color, output, error, success)
    assert isinstance(result, ColoramaPrinter)
    assert result.output == sys.stdout
    assert result.error_message == error
    assert result.success_message == success

def test_create_terminal_printer_without_color():
    color = False
    output = None
    error = "Error: {error} - {message}"
    success = "Success: {success} - {message}"
    result = create_terminal_printer(color, output, error, success)
    assert isinstance(result, BasicPrinter)
    assert result.output == sys.stdout
    assert result.error_message == error
    assert result.success_message == success

def test_create_terminal_printer_with_custom_output():
    color = False
    output = io.StringIO()
    error = "Error: {error} - {message}"
    success = "Success: {success} - {message}"
    result = create_terminal_printer(color, output, error, success)
    assert isinstance(result, BasicPrinter)
    assert result.output == output
    assert result.error_message == error
    assert result.success_message == success


# LLM-generated content at query #2
#--------------------------

```python
def test_create_terminal_printer_with_color_and_colorama_unavailable():
    colorama_unavailable = True
    with pytest.raises(SystemExit) as excinfo:
        create_terminal_printer(color=True, colorama_unavailable=colorama_unavailable)
    assert excinfo.value.code == 1


# LLM-generated content at query #3
#--------------------------

```python
def test_create_terminal_printer_colorama_available():
    colorama_unavailable = False
    result = create_terminal_printer(color=True)
    assert isinstance(result, ColoramaPrinter)


