####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_create_terminal_printer_no_color():
    printer = create_terminal_printer(color=False)
    assert isinstance(printer, BasicPrinter)
    assert printer.output == sys.stdout
    assert printer.error_message == ""
    assert printer.success_message == ""

def test_create_terminal_printer_with_color():
    printer = create_terminal_printer(color=True)
    assert isinstance(printer, ColoramaPrinter)
    assert printer.output == sys.stdout
    assert printer.error_message == ""
    assert printer.success_message == ""

def test_create_terminal_printer_custom_messages():
    printer = create_terminal_printer(color=False, error="Custom Error: {error}", success="Custom Success: {success}")
    assert isinstance(printer, BasicPrinter)
    assert printer.error_message == "Custom Error: {error}"
    assert printer.success_message == "Custom Success: {success}"

def test_create_terminal_printer_custom_output():
    output = io.StringIO()
    printer = create_terminal_printer(color=False, output=output)
    assert isinstance(printer, BasicPrinter)
    assert printer.output == output


# LLM-generated content at query #2
#--------------------------

```python
def test_create_terminal_printer_colorama_unavailable():
    colorama_unavailable = True
    color = True
    output = None
    error = ""
    success = ""

    with pytest.raises(SystemExit) as excinfo:
        create_terminal_printer(color, output, error, success)
    assert excinfo.value.code == 1


# LLM-generated content at query #3
#--------------------------

```python
def test_colorama_unavailable_with_color_exits():
    with pytest.raises(SystemExit) as exit_info:
        create_terminal_printer(color=True, output=None, error="", success="")
    assert exit_info.value.code == 1


