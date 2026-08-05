####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    # Mocking the function object provided by cssselect
    mock_function = MagicMock()
    
    # Case 1: Valid numeric argument (e.g., :lt(1) should result in position() < 2)
    mock_argument = MagicMock()
    mock_argument.value = '1'
    mock_function.arguments = [mock_argument]
    mock_function.argument_types.return_value = ['NUMBER']
    
    result = translator.xpath_lt_function(xpath_expr, mock_function)
    assert result == xpath_expr
    # XPath position is 1-indexed, so :lt(1) means index < 2 (which is index 1)
    assert 'position() < 2' in str(xpath_expr)

    # Case 2: Invalid argument type (e.g., passing a STRING instead of NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath_expr, mock_function)

    # Case 3: Another valid numeric argument (e.g., :lt(0) should result in position() < 1)
    mock_argument.value = '0'
    mock_function.arguments = [mock_argument]
    mock_function.argument_types.return_value = ['NUMBER']
    
    translator.xpath_lt_function(xpath_expr, mock_function)
    assert 'position() < 1' in str(xpath_expr)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_arg_value = MagicMock()
    mock_arg_value.value = '0'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg_value]
    
    # Test valid input: :eq(0) should result in position() = 1
    result = translator.xpath_eq_function(xpath, mock_function)
    assert result == xpath
    assert 'position() = 1' in str(xpath)
    
    # Test invalid argument type: STRING instead of NUMBER
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock(value='"0"')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_function)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Test valid input with different index: :eq(2) should result in position() = 3
    mock_function.argument_types.return_value = ['NUMBER']
    mock_arg_value_2 = MagicMock()
    mock_arg_value_2.value = '2'
    mock_function.arguments = [mock_arg_value_2]
    
    xpath_new = XPathExpr(path='p')
    translator.xpath_eq_function(xpath_new, mock_function)
    assert 'position() = 3' in str(xpath_new)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')

    # Mocking the function object for a valid numeric argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '1'
    mock_func_valid.arguments = [mock_arg_valid]

    # Test valid input: :lt(1) should result in position() < 2
    result_xpath = translator.xpath_lt_function(xpath, mock_func_valid)
    assert str(result_xpath) == "div[position() < 2]"
    assert result_xpath is xpath

    # Test invalid input: non-numeric argument type should raise ExpressionError
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='abc')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, mock_func_invalid)
    assert "Expected a single integer" in str(excinfo.value)

    # Test edge case: index 0 (the first element)
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_func_zero.arguments = [mock_arg_zero]
    
    xpath_new = XPathExpr(path='p')
    translator.xpath_lt_function(xpath_new, mock_func_zero)
    assert str(xpath_new) == "p[position() < 1]"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_arg = MagicMock()
    mock_arg.value = '1'
    
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    # Test valid input: :lt(1) should result in position() < 2
    result = translator.xpath_lt_function(xpath, mock_function)
    assert result == xpath
    assert 'position() < 2' in str(xpath)

    # Test invalid argument type (e.g., STRING)
    mock_arg_bad = MagicMock()
    mock_arg_bad.value = 'abc'
    
    mock_function_bad = MagicMock()
    mock_function_bad.argument_types = ['STRING']
    mock_function_bad.arguments = [mock_arg_bad]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, mock_function_bad)
    assert "Expected a single integer for :gt(), got" in str(excinfo.value) or \
           "Expected a single integer for :lt(), got" in str(excinfo.value)

    # Test with zero index: :lt(0) should result in position() < 1 (which is impossible in XPath 1-based)
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types = ['NUMBER']
    mock_function_zero.arguments = [mock_arg_zero]
    
    xpath_new = XPathExpr(path='p')
    translator.xpath_lt_function(xpath_new, mock_function_zero)
    assert 'position() < 1' in str(xpath_new)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mocking XPathExpr (XPathExprOrig)
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input - integer index
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_argument_valid = MagicMock()
    mock_argument_valid.value = '0'
    mock_function_valid.arguments = [mock_argument_valid]
    
    translator.xpath_eq_function(xpath_expr, mock_function_valid)
    # Index 0 in JS/jQuery maps to position() = 1 in XPath
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
    
    # Case 2: Valid input - higher index
    mock_function_valid_high = MagicMock()
    mock_function_valid_high.argument_types.return_value = ['NUMBER']
    mock_argument_high = MagicMock()
    mock_argument_high.value = '5'
    mock_function_valid_high.arguments = [mock_argument_high]
    
    translator.xpath_eq_function(xpath_expr, mock_function_valid_high)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')

    # Case 3: Invalid input - wrong argument type (e.g., STRING)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='abc')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid NUMBER argument (e.g., :gt(0))
    # In XPath, position is 1-indexed, so :gt(0) should result in position() > 1
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_function_valid.arguments = [mock_arg]
    
    result = translator.xpath_gt_function(xpath, mock_function_valid)
    
    assert result == xpath
    xpath.add_post_condition.assert_called_with('position() > 1')
    
    # Case 2: Invalid argument type (e.g., passing a string instead of number)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='abc')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Case 3: Testing with a different index (e.g., :gt(2))
    mock_function_two = MagicMock()
    mock_function_two.argument_types.return_value = ['NUMBER']
    mock_arg_two = MagicMock()
    mock_arg_two.value = '2'
    mock_function_two.arguments = [mock_arg_two]
    
    translator.xpath_gt_function(xpath, mock_function_two)
    xpath.add_post_condition.assert_called_with('position() > 3')
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed to xpath_contains_function
    # It needs an 'arguments' attribute with a '.value' attribute
    mock_arg = MagicMock()
    mock_arg.value = 'target_text'
    
    mock_function = MagicMock()
    mock_function.arguments = [mock_arg]
    # Case 1: Valid STRING type
    mock_function.argument_types.return_value = ['STRING']

    # Execute the function
    result = translator.xpath_contains_function(xpath_expr, mock_function)

    # Verify the return value is the xpath object itself (chainable)
    assert result == xpath_expr
    
    # Verify the post condition was added correctly
    # Note: xpath_literal wraps the string in quotes for XPath
    expected_post_condition = "contains(., 'target_text')"
    assert 'contains(., \'target_text\')' in str(xpath_expr)

    # Case 2: Valid IDENT type
    mock_function.argument_types.return_value = ['IDENT']
    xpath_expr_ident = XPathExpr(path='div')
    translator.xpath_contains_function(xpath_expr_ident, mock_function)
    assert 'contains(., \'target_text\')' in str(xpath_expr_ident)

    # Case 3: Invalid type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    xpath_expr_error = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_error, mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Case 4: Invalid type (e.g., BOOLEAN) should raise ExpressionError
    mock_function.argument_types.return_value = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr_error, mock_function)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='p')
    
    # Mock function for valid input: :lt(1) -> position() < 2
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [MagicMock(value='1')]
    
    xpath_result = translator.xpath_lt_function(xpath, mock_func_valid)
    assert xpath_result == xpath
    assert 'position() < 2' in str(xpath)

    # Mock function for invalid input type: :lt("string")
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='abc')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :gt(), got" in str(excinfo.value) or "Expected a single integer" in str(excinfo.value)

    # Test with zero index: :lt(0) -> position() < 1 (which is impossible in XPath 1-based indexing)
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_func_zero.arguments = [MagicMock(value='0')]
    
    xpath_zero = XPathExpr(path='div')
    translator.xpath_lt_function(xpath_zero, mock_func_zero)
    assert 'position() < 1' in str(xpath_zero)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Create a mock function object with argument properties
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    
    # Mock the argument value (e.g., :lt(1) -> index 0)
    mock_arg = MagicMock()
    mock_arg.value = '1'
    mock_function.arguments = [mock_arg]

    # Test Case 1: Valid NUMBER input
    # For :lt(1), the logic is position() < (1 + 1) => position() < 2
    translator.xpath_lt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() < 2')

    # Test Case 2: Invalid argument type (e.g., STRING)
    mock_function.argument_types = ['STRING']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function)
    assert "Expected a single integer for :gt(), got" in str(excinfo.value)

    # Test Case 3: Verify functionality with different numeric value
    mock_function.argument_types = ['NUMBER']
    mock_arg.value = '5'
    mock_function.arguments = [mock_arg]
    xpath_expr.reset_mock()
    
    translator.xpath_lt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() < 6')
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mocking the function object passed to xpath_has_function
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument (e.g., :has(".bar"))
    mock_func.arguments = [MagicMock(value='.bar')]
    # We need to mock css_to_xpath because it's called within xpath_has_function
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')
    
    result = translator.xpath_has_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert 'descendant::*[@class="bar"]' in str(xpath_expr)
    
    # Reset for next case
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Case 2: Valid IDENT argument (e.g., :has(div))
    mock_func.arguments = [MagicMock(value='div')]
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert 'descendant::div' in str(xpath_expr)

    # Case 3: Invalid argument type (e.g., :has(123) where types are not STRING or IDENT)
    mock_func.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, mock_func)
    assert "Expected a single string or ident for :has()" in str(excinfo.value)

    # Case 4: Another invalid argument type (e.g., :has(true))
    mock_func.argument_types.return_value = ['BOOLEAN']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, mock_func)
    assert "Expected a single string or ident for :has()" in str(excinfo.value)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function argument
    def create_mock_arg(value, arg_types):
        arg = MagicMock()
        arg.value = value
        arg.argument_types = arg_types
        return arg

    # Mocking css_to_xpath behavior
    translator.css_to_xpath = MagicMock(side_effect=lambda selector, prefix='': f"{prefix}{selector}")

    # Test Case 1: Valid STRING argument (e.g., :has(".bar"))
    func_str = MagicMock()
    func_str.arguments = [create_mock_arg('.bar', 'STRING')]
    translator.xpath_has_function(xpath, func_str)
    xpath.add_post_condition.assert_called_with('descendant::.bar')
    
    # Reset mock for next case
    xpath.add_post_condition.reset_mock()

    # Test Case 2: Valid IDENT argument (e.g., :has(div))
    func_ident = MagicMock()
    func_ident.arguments = [create_mock_arg('div', 'IDENT')]
    translator.xpath_has_function(xpath, func_ident)
    xpath.add_post_condition.assert_called_with('descendant::div')

    # Reset mock for next case
    xpath.add_post_condition.reset_mock()

    # Test Case 3: Invalid argument type (e.g., :has(123) where number is not allowed)
    func_invalid = MagicMock()
    func_invalid.arguments = [create_mock_arg('123', 'NUMBER')]
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath, func_invalid)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Test Case 4: Empty arguments (should also trigger error)
    func_empty = MagicMock()
    func_empty.arguments = []
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, func_empty)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mock function object for :has()
    mock_function = MagicMock()
    # Simulate argument: .bar is a string literal in the context of CSS/XPath processing
    arg = MagicMock()
    arg.value = '.bar'
    mock_function.arguments = [arg]
    mock_function.argument_types = ['STRING']

    # We need to mock css_to_xpath because it's an inherited method 
    # and we want to control the output to test the post-condition logic
    translator.css_to_xpath = MagicMock(return_value='descendant::*[contains(concat("class=", @class), " bar ")]')

    # Execute the function
    result = translator.xpath_has_function(xpath_expr, mock_function)

    # Assertions
    assert result == xpath_expr
    assert 'descendant::*[contains(concat("class=", @class), " bar ")]' in str(xpath_expr)

    # Test for ExpressionError with wrong argument type (e.g., NUMBER instead of STRING)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr, mock_function)

    # Test with IDENT type (valid according to code)
    mock_function.argument_types = ['IDENT']
    arg_ident = MagicMock()
    arg_ident.value = 'div'
    mock_function.arguments = [arg_ident]
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result_ident = translator.xpath_has_function(xpath_expr, mock_function)
    assert 'descendant::div' in str(result_ident)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed by cssselect/pyquery
    mock_function = MagicMock()
    
    # Test Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock(value='"hello"')]
    # Mocking xpath_literal to return the value as is for simplicity in test
    translator.xpath_literal = MagicMock(side_effect=lambda x: x)
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    assert result == xpath_expr
    assert 'contains(., "hello")' in str(xpath_expr)

    # Test Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments = [MagicMock(value='some_id')]
    translator.xpath_literal = MagicMock(side_effect=lambda x: f"'{x}'")
    
    xpath_expr_2 = XPathExpr(path='div')
    result_2 = translator.xpath_contains_function(xpath_expr_2, mock_function)
    assert result_2 == xpath_expr_2
    assert "contains(., 'some_id')" in str(xpath_expr_2)

    # Test Case 3: Invalid argument type (NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock(value='123')]
    
    xpath_expr_3 = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_3, mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Test Case 4: Invalid argument type (LIST/Complex) should raise ExpressionError
    mock_function.argument_types.return_value = ['LIST']
    xpath_expr_4 = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_4, mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock the function object passed to the translator
    mock_func = MagicMock()
    # Create a mock argument with a value
    mock_arg = MagicMock()
    mock_arg.value = 'target_text'
    mock_func.arguments = [mock_arg]
    
    # Test Case 1: Valid STRING type
    mock_func.argument_types.return_value = ['STRING']
    # We mock cssselect_xpath.xpath_literal via the translator if necessary, 
    # but here we assume it's part of the base class and works or is mocked.
    # For this unit test, let's ensure the logic flows to post_condition.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(translator, 'xpath_literal', lambda x: f"'{x}'")
        result = translator.xpath_contains_function(xpath_expr, mock_func)
        assert result == xpath_expr
        assert "contains(., 'target_text')" in str(xpath_expr)

    # Test Case 2: Valid IDENT type
    mock_func.argument_types.return_value = ['IDENT']
    xpath_expr_ident = XPathExpr(path='div')
    result_ident = translator.xpath_contains_function(xpath_expr_ident, mock_func)
    assert "contains(., 'target_text')" in str(xpath_expr_ident)

    # Test Case 3: Invalid type (e.g., NUMBER) should raise ExpressionError
    mock_func.argument_types.return_value = ['NUMBER']
    xpath_expr_error = XPathExpr(path='div')
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr_error, mock_func)

    # Test Case 4: Invalid type (e.g., BOOLEAN) should raise ExpressionError
    mock_func.argument_types.return_value = ['BOOLEAN']
    xpath_expr_error_bool = XPathExpr(path='div')
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr_error_bool, mock_func)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mocking XPathExpr (the xpath argument)
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function object for the CSS function
    def create_mock_function(arg_types, arg_value):
        mock_func = MagicMock()
        mock_func.argument_types.return_value = arg_types
        mock_arg = MagicMock()
        mock_arg.value = arg_value
        mock_func.arguments = [mock_arg]
        return mock_func

    # Test Case 1: Valid integer argument (0-indexed)
    # :eq(0) should result in position() = 1
    func_valid = create_mock_function(['NUMBER'], '0')
    result = translator.xpath_eq_function(xpath_expr, func_valid)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('position() = 1')

    # Test Case 2: Valid integer argument (1-indexed)
    # :eq(5) should result in position() = 6
    func_valid_large = create_mock_function(['NUMBER'], '5')
    translator.xpath_eq_function(xpath_expr, func_valid_large)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')

    # Test Case 3: Invalid argument type (STRING instead of NUMBER)
    # Should raise ExpressionError
    func_invalid = create_mock_function(['STRING'], '"0"')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, func_invalid)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Test Case 4: Invalid argument type (IDENT instead of NUMBER)
    func_invalid_ident = create_mock_function(['IDENT'], 'foo')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, func_invalid_ident)
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function argument
    def create_mock_arg(value, arg_types):
        arg = MagicMock()
        arg.value = value
        return arg

    # Test Case 1: Valid STRING argument
    func_string = MagicMock()
    func_string.argument_types = ['STRING']
    func_string.arguments = [create_mock_arg('.bar', 'STRING')]
    
    # Mock css_to_xpath to return a specific xpath string
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')
    
    result = translator.xpath_has_function(xpath_expr, func_string)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::*[@class="bar"]')

    # Test Case 2: Valid IDENT argument
    func_ident = MagicMock()
    func_ident.argument_types = ['IDENT']
    func_ident.arguments = [create_mock_arg('div', 'IDENT')]
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, func_ident)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::div')

    # Test Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    func_invalid = MagicMock()
    func_invalid.argument_types = ['NUMBER']
    func_invalid.arguments = [create_mock_arg('123', 'NUMBER')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, func_invalid)
    
    assert "Expected a single string or ident for :has()" in str(excinfo.value)

    # Test Case 4: Empty/None arguments check (edge case for type validation)
    func_empty = MagicMock()
    func_empty.argument_types = []
    func_empty.arguments = []
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr, func_empty)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '1'
    mock_func_valid.arguments = [mock_arg_valid]
    
    # Test valid :gt(1) -> position() > 2
    result = translator.xpath_gt_function(xpath_expr, mock_func_valid)
    assert result == xpath_expr
    assert 'position() > 2' in str(xpath_expr)
    
    # Mock for an invalid argument type (e.g., STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='foo')]
    
    # Test that passing a non-NUMBER type raises ExpressionError
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test with zero index :gt(0) -> position() > 1
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_func_zero.arguments = [mock_arg_zero]
    
    xpath_expr_zero = XPathExpr(path='p')
    translator.xpath_gt_function(xpath_expr_zero, mock_func_zero)
    assert 'position() > 1' in str(xpath_expr_zero)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '1'
    mock_func_valid.arguments = [mock_arg_valid]
    
    # Test successful execution: :lt(1) should result in position() < 2
    result = translator.xpath_lt_function(xpath_expr, mock_func_valid)
    assert result == xpath_expr
    assert 'position() < 2' in str(xpath_expr)

    # Mock for an invalid argument type (e.g., STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='abc')]
    
    # Test that ExpressionError is raised for non-NUMBER types
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath_expr, mock_func_invalid)

    # Reset xpath_expr for clean state testing
    xpath_expr_reset = XPathExpr(path='p')
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_func_zero.arguments = [mock_arg_zero]

    # Test :lt(0) should result in position() < 1
    translator.xpath_lt_function(xpath_expr_reset, mock_func_zero)
    assert 'position() < 1' in str(xpath_expr_reset)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '0'
    mock_func_valid.arguments = [mock_arg_valid]

    # Test valid :gt(0) -> position() > 1
    result_valid = translator.xpath_gt_function(xpath, mock_func_valid)
    assert result_valid == xpath
    assert 'position() > 1' in str(xpath)

    # Mock for an invalid argument type (STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='abc')]

    # Test error raising on invalid type
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test another valid value :gt(2) -> position() > 3
    xpath_new = XPathExpr(path='p')
    mock_func_val2 = MagicMock()
    mock_func_val2.argument_types.return_value = ['NUMBER']
    mock_arg_val2 = MagicMock()
    mock_arg_val2.value = '2'
    mock_func_val2.arguments = [mock_arg_val2]

    result_val2 = translator.xpath_gt_function(xpath_new, mock_func_val2)
    assert 'position() > 3' in str(xpath_new)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # 1. Test valid input: integer index (0-indexed in jQuery, becomes position() = value + 1 in XPath)
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    
    # Mock the argument structure used by cssselect/xpath
    arg_0 = MagicMock()
    arg_0.value = '1'
    mock_func_valid.arguments = [arg_0]
    
    translator.xpath_eq_function(mock_xpath, mock_func_valid)
    mock_xpath.add_post_condition.assert_called_once_with('position() = 2')
    
    # Reset mock for next scenario
    mock_xpath.reset_mock()

    # 2. Test invalid input: non-NUMBER argument type (should raise ExpressionError)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [arg_0]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
    mock_xpath.add_post_condition.assert_not_called()

    # 3. Test valid input: zero index
    arg_zero = MagicMock()
    arg_zero.value = '0'
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_func_zero.arguments = [arg_zero]

    translator.xpath_eq_function(mock_xpath, mock_func_zero)
    mock_xpath.add_post_condition.assert_called_with('position() = 1')
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    mock_xpath = MagicMock()
    
    # Case 1: Valid input (NUMBER)
    # Create a mock function object with argument types and values
    mock_func = MagicMock()
    mock_func.argument_types.return_value = ['NUMBER']
    
    # Mock the argument structure of cssselect's XPath function
    mock_arg = MagicMock()
    mock_arg.value = '1'
    mock_func.arguments = [mock_arg]
    
    # Execution: :lt(1) should result in position() < 2 (since index is 0-based and xpath is 1-based)
    result = translator.xpath_lt_function(mock_xpath, mock_func)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('position() < 2')
    
    # Case 2: Invalid input (Not a NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='foo')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single integer for :gt(), got" in str(excinfo.value) or \
           "Expected a single integer for :lt(), got" in str(excinfo.value)

    # Case 3: Verify math logic with different number
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_func_zero.arguments = [mock_arg_zero]
    
    translator.xpath_lt_function(mock_xpath, mock_func_zero)
    # :lt(0) means position < 1
    mock_xpath.add_post_condition.assert_called_with('position() < 1')
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function argument object used by cssselect
    class MockArgument:
        def __init__(self, value):
            self.value = value

    class MockFunction:
        def __init__(self, arg_types, arguments):
            self.argument_types = arg_types
            self.arguments = arguments

    # Test Case 1: Valid integer input (0-indexed in jQuery, 1-indexed in XPath)
    arg_val = MockArgument('0')
    func_valid = MockFunction(['NUMBER'], [arg_val])
    result_valid = translator.xpath_eq_function(xpath, func_valid)
    assert result_valid == xpath
    assert 'position() = 1' in str(xpath)

    # Test Case 2: Valid integer input (higher index)
    arg_val_high = MockArgument('5')
    func_high = MockFunction(['NUMBER'], [arg_val_high])
    translator.xpath_eq_function(xpath, func_high)
    assert 'position() = 6' in str(xpath)

    # Test Case 3: Invalid argument type (STRING instead of NUMBER)
    arg_str = MockArgument('abc')
    func_invalid_type = MockFunction(['STRING'], [arg_str])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, func_invalid_type)
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)

    # Test Case 4: Invalid argument type (IDENT instead of NUMBER)
    arg_ident = MockArgument('some_id')
    func_invalid_ident = MockFunction(['IDENT'], [arg_ident])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, func_invalid_ident)
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid integer argument
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '0'
    
    result = translator.xpath_gt_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('position() > 1')
    
    # Case 2: Valid integer argument (different value)
    mock_function.arguments[0].value = '5'
    translator.xpath_gt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() > 6')

    # Case 3: Invalid argument type (e.g., STRING)
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'abc'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., IDENT)
    mock_function.argument_types.return_value = ['IDENT']
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath_expr, mock_function)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object passed to xpath_lt_function
    # We need to simulate an argument with a .value attribute
    mock_arg = MagicMock()
    mock_arg.value = '1'
    
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [mock_arg]

    # Test Case 1: Valid input (lt(1) should result in position() < 2)
    result_xpath = translator.xpath_lt_function(xpath, mock_function)
    assert 'position() < 2' in str(result_xpath)
    assert result_xpath is xpath

    # Test Case 2: Invalid argument type (should raise ExpressionError)
    mock_function.argument_types = ['STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(XPathExpr('div'), mock_function)

    # Test Case 3: Valid input with different value (lt(0) should result in position() < 1)
    mock_arg.value = '0'
    mock_function.argument_types = ['NUMBER']
    new_xpath = XPathExpr('div')
    translator.xpath_lt_function(new_xpath, mock_function)
    assert 'position() < 1' in str(new_xpath)
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object provided by cssselect
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.argument_types.return_value = ['STRING']
    mock_func.arguments = [MagicMock()]
    mock_func.arguments[0].value = 'hello'
    
    # Mock cssselect literal formatting behavior if needed, 
    # but here we assume it returns a string compatible with XPath
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert "contains(., 'hello')" in str(xpath_expr)
    
    # Case 2: Valid IDENT argument
    mock_func.argument_types.return_value = ['IDENT']
    mock_func.arguments[0].value = 'title'
    translator.xpath_literal = MagicMock(return_value='title')
    
    new_expr = XPathExpr(path='div')
    translator.xpath_contains_function(new_expr, mock_func)
    assert "contains(., title)" in str(new_expr)

    # Case 3: Invalid argument type (NUMBER)
    mock_func.argument_types.return_value = ['NUMBER']
    mock_func.arguments = [MagicMock()]
    mock_func.arguments[0].value = 123
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Case 4: Invalid argument type (LIST/Other)
    mock_func.argument_types.return_value = ['LIST']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident" in str(excinfo.value)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mocking XPathExpr (which inherits from XPathExprOrig)
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input - integer argument
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    translator.xpath_gt_function(xpath_expr, mock_function)
    # :gt(0) should result in position() > 1 (since it's 0-indexed in jQuery but 1-indexed in XPath)
    xpath_expr.add_post_condition.assert_called_once_with('position() > 1')
    
    # Reset mock for next case
    xpath_expr.add_post_condition.reset_mock()

    # Case 2: Valid input - different integer argument
    mock_arg.value = '5'
    translator.xpath_gt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_once_with('position() > 6')

    # Case 3: Invalid input - wrong argument type (e.g., STRING)
    mock_function.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Case 4: Invalid input - empty argument types
    mock_function.argument_types.return_value = []
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    # Setup translator and mock xpath expression
    translator = JQueryTranslator()
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Mock function object for :has(".bar")
    # Argument is a string literal (type STRING) containing ".bar"
    mock_function = MagicMock()
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    mock_function.arguments = [mock_arg]
    mock_function.argument_types = ['STRING']

    # Mock css_to_xpath to return the expected xpath fragment
    # For :has(".bar"), it should prepend descendant::
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')

    # Execute the function under test
    result = translator.xpath_has_function(mock_xpath, mock_function)

    # Verify that add_post_condition was called with the correct transformed value
    mock_xpath.add_post_condition.assert_called_once_with('descendant::*[@class="bar"]')
    
    # Ensure it returns the xpath object itself (chainable)
    assert result == mock_xpath

    # Test Error Case: Invalid argument type (e.g., NUMBER instead of STRING/IDENT)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(mock_xpath, mock_function)

    # Test Success Case with IDENT type
    mock_function.argument_types = ['IDENT']
    mock_arg.value = 'div'
    translator.css_to_xpath.return_value = 'descendant::div'
    
    result_ident = translator.xpath_has_function(mock_xpath, mock_function)
    mock_xpath.add_post_condition.assert_called_with('descendant::div')
    assert result_ident == mock_xpath

    # Test Error Case: Invalid argument type (e.g., BOOLEAN)
    mock_function.argument_types = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(mock_xpath, mock_function)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_last_pseudo():
    # Arrange
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    # Act
    result = translator.xpath_last_pseudo(xpath_expr)
    
    # Assert
    assert result is xpath_expr
    assert xpath_expr.post_condition == 'position() = last()'
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument
    # We need to mock function.argument_types and function.arguments[0].value
    mock_function_str = MagicMock()
    mock_function_str.argument_types = ['STRING']
    mock_arg_str = MagicMock()
    mock_arg_str.value = '.baz'
    mock_function_str.arguments = [mock_arg_str]
    
    # Mock css_to_xpath to return the expected descendant path
    translator.css_to_xpath = MagicMock(return_value='descendant::.baz')
    
    result = translator.xpath_has_function(xpath_expr, mock_function_str)
    
    # Assertions for Case 1
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::.baz')

    # Case 2: Valid IDENT argument
    mock_function_ident = MagicMock()
    mock_function_ident.argument_types = ['IDENT']
    mock_arg_ident = MagicMock()
    mock_arg_ident.value = 'div'
    mock_function_ident.arguments = [mock_arg_ident]
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, mock_function_ident)
    
    # Assertions for Case 2
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types = ['NUMBER']
    mock_function_invalid.arguments = [MagicMock(value=1)]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single string or ident for :has(), got" in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # 1. Test valid input: :gt(0) -> position() > 1
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_argument_valid = MagicMock()
    mock_argument_valid.value = '0'
    mock_function_valid.arguments = [mock_argument_valid]
    
    translator.xpath_gt_function(xpath_expr, mock_function_valid)
    xpath_expr.add_post_condition.assert_called_with('position() > 1')

    # 2. Test valid input: :gt(5) -> position() > 6
    mock_function_valid.arguments = [MagicMock(value='5')]
    translator.xpath_gt_function(xpath_expr, mock_function_valid)
    xpath_expr.add_post_condition.assert_called_with('position() > 6')

    # 3. Test invalid input type (e.g., STRING instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='"abc"')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_JQueryTranslator_xpath_first_pseudo():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    # Apply the pseudo-class transformation
    result = translator.xpath_first_pseudo(xpath_expr)
    
    # Verify that the returned object is the same instance
    assert result is xpath_expr
    
    # Verify that the post_condition was added correctly
    assert xpath_expr.post_condition == 'position() = 1'
    
    # Verify the string representation includes the new condition in brackets
    # Note: XPathExpr inherits from XPathExprOrig, which handles the base path.
    # The custom __str__ implementation of XPathExpr adds [condition]
    assert str(xpath_expr) == 'p[position() = 1]'

def test_JQueryTranslator_xpath_first_pseudo_with_existing_post_condition():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    xpath_expr.add_post_condition('class="container"')
    
    translator.xpath_first_pseudo(xpath_expr)
    
    # Ensure it appends or maintains the structure correctly via add_post_condition logic
    # In JQueryTranslator, xpath_first_pseudo calls add_post_condition directly
    assert 'position() = 1' in str(xpath_expr)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_password_pseudo():
    # Arrange
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='input')
    
    # Act
    result = translator.xpath_password_pseudo(xpath_expr)
    
    # Assert
    assert result is xpath_expr
    assert xpath_expr.condition == "@type = 'password' and name(.) = 'input'"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid integer argument (index 0 -> position() = 1)
    mock_arg_0 = MagicMock()
    mock_arg_0.value = '0'
    
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_function_valid.arguments = [mock_arg_0]
    
    translator.xpath_eq_function(xpath_expr, mock_function_valid)
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
    
    # Case 2: Valid integer argument (index 5 -> position() = 6)
    mock_arg_5 = MagicMock()
    mock_arg_5.value = '5'
    
    mock_function_valid_5 = MagicMock()
    mock_function_valid_5.argument_types.return_value = ['NUMBER']
    mock_function_valid_5.arguments = [mock_arg_5]
    
    translator.xpath_eq_function(xpath_expr, mock_function_valid_5)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')

    # Case 3: Invalid argument type (STRING instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_0]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_invalid)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Case 4: Invalid argument type (IDENT instead of NUMBER)
    mock_function_ident = MagicMock()
    mock_function_ident.argument_types.return_value = ['IDENT']
    mock_function_ident.arguments = [mock_arg_0]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_ident)
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock the function object provided by cssselect
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'hello'
    
    # Mock xpath_literal to return a formatted XPath string
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    assert "contains(., 'hello')" in str(xpath_expr)

    # Case 2: Valid IDENT argument (e.g., a class or unquoted string)
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'some_ident'
    translator.xpath_literal = MagicMock(return_value='some_ident')
    
    # Reset expr for fresh test
    new_expr = XPathExpr(path='div')
    translator.xpath_contains_function(new_expr, mock_function)
    assert "contains(., some_ident)" in str(new_expr)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)

    # Case 4: Invalid argument type (e.g., LIST/other) should raise ExpressionError
    mock_function.argument_types = ['LIST']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input (NUMBER type)
    # Create a mock function object with argument properties
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    
    # Mock the value of the first argument
    mock_arg = MagicMock()
    mock_arg.value = '1'
    mock_function.arguments = [mock_arg]
    
    result = translator.xpath_lt_function(xpath_expr, mock_function)
    
    # Verify result is the xpath object itself (fluent interface)
    assert result == xpath_expr
    # Verify add_post_condition was called with correct XPath logic: position() < value + 1
    # Since input is 1, expected is position() < 2
    xpath_expr.add_post_condition.assert_called_with('position() < 2')

    # Case 2: Invalid input type (e.g., STRING)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types = ['STRING']
    mock_function_invalid.arguments = [mock_arg]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value) or \
           "Expected a single integer for :lt()" in str(excinfo.value)

    # Case 3: Another valid input (0 index)
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero.arguments = [mock_arg_zero]
    
    translator.xpath_lt_function(xpath_expr, mock_function_zero)
    xpath_expr.add_post_condition.assert_called_with('position() < 1')
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Mock the function argument (CSS selector inside :has())
    mock_arg = MagicMock()
    mock_arg.value = '.baz'
    
    function = MagicMock()
    function.arguments = [mock_arg]
    function.argument_types.return_value = ['STRING']
    
    # Mock css_to_xpath to return the expected descendant axis xpath
    translator.css_to_xpath = MagicMock(return_value='descendant::.baz')

    # Test Case 1: Valid STRING argument
    result = translator.xpath_has_function(xpath_expr, function)
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::.baz')

    # Test Case 2: Valid IDENT argument
    function.argument_types.return_value = ['IDENT']
    result = translator.xpath_has_function(xpath_expr, function)
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::.baz')

    # Test Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, function)
    assert "Expected a single string or ident for :has()" in str(excinfo.value)

    # Test Case 4: Another valid IDENT argument with different selector
    mock_arg.value = 'div'
    function.argument_types.return_value = ['IDENT']
    translator.css_to_xpath.return_value = 'descendant::div'
    
    result = translator.xpath_has_function(xpath_expr, function)
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::div')
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    # Initialize translator and mock XPathExpr
    translator = JQueryTranslator()
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Mock the function argument for :contains("text")
    mock_function = MagicMock()
    mock_arg = MagicMock()
    mock_arg.value = 'target_text'
    mock_function.arguments = [mock_arg]
    mock_function.argument_types = ['STRING']

    # Mock cssselect_xpath.XPathExpr.__str__ or similar if needed, 
    # but here we primarily test the logic of add_post_condition call.
    # We need to mock xpath_literal because it's called by xpath_contains_function
    translator.xpath_literal = MagicMock(return_value="'target_text'")

    # Execute the function
    result = translator.xpath_contains_function(xpath_expr, mock_function)

    # Verify that add_post_condition was called with the correctly formatted XPath string
    xpath_expr.add_post_condition.assert_called_once_with("contains(., 'target_text')")
    assert result == xpath_expr

    # Test for ExpressionError when argument type is invalid (e.g., NUMBER instead of STRING)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr, mock_function)

    # Test with IDENT type (valid per implementation)
    mock_function.argument_types = ['IDENT']
    translator.xpath_literal = MagicMock(return_value='"target_text"')
    translator.xpath_contains_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with("contains(., \"target_text\")")

    # Test for ExpressionError when argument type is completely wrong
    mock_function.argument_types = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr, mock_function)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mocking the function object passed to xpath_has_function
    # We need to simulate a function argument with a value and type
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    
    mock_function = MagicMock()
    mock_function.arguments = [mock_arg]
    mock_function.argument_types = ['STRING']

    # Mock css_to_xpath to return a specific string for the test
    translator.css_to_xpath = MagicMock(return_value='descendant::*.bar')

    # Test successful execution
    result = translator.xpath_has_function(xpath_expr, mock_function)
    assert result == xpath_expr
    # Check if post_condition was added correctly via the descendant prefix logic
    assert 'descendant::*.bar' in str(xpath_expr)

    # Test with IDENT type (which is also allowed according to code)
    mock_arg.value = 'div'
    mock_function.argument_types = ['IDENT']
    translator.css_to_xpath.return_value = 'descendant::div'
    
    new_xpath = XPathExpr(path='*', element='*')
    translator.xpath_has_function(new_xpath, mock_function)
    assert 'descendant::div' in str(new_xpath)

    # Test with invalid argument type (should raise ExpressionError)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(XPathExpr(), mock_function)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Test with an empty arguments list (should raise ExpressionError)
    mock_function.arguments = []
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed to xpath_contains_function
    mock_func = MagicMock()
    
    # 1. Test successful execution with STRING argument type
    mock_func.argument_types.return_value = ['STRING']
    mock_func.arguments = [MagicMock()]
    mock_func.arguments[0].value = 'hello'
    
    # Mock xpath_literal to return a formatted XPath string
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert "contains(., 'hello')" in str(xpath_expr)

    # 2. Test successful execution with IDENT argument type
    mock_func.argument_types.return_value = ['IDENT']
    translator.xpath_literal = MagicMock(return_value='myident')
    
    xpath_expr_2 = XPathExpr(path='p')
    translator.xpath_contains_function(xpath_expr_2, mock_func)
    assert "contains(., myident)" in str(xpath_expr_2)

    # 3. Test error handling for invalid argument type (e.g., NUMBER)
    mock_func.argument_types.return_value = ['NUMBER']
    xpath_expr_err = XPathExpr(path='span')
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_err, mock_func)
    
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument (e.g., :has(".bar"))
    arg_string = MagicMock()
    arg_string.value = ".bar"
    function_string = MagicMock()
    function_string.argument_types = ['STRING']
    function_string.arguments = [arg_string]
    
    # Mocking css_to_xpath to return a dummy xpath string
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')
    
    result = translator.xpath_has_function(xpath_expr, function_string)
    
    # Verify that add_post_condition was called with the correctly prefixed xpath
    xpath_expr.add_post_condition.assert_called_with('descendant::*[@class="bar"]')
    assert result == xpath_expr

    # Case 2: Valid IDENT argument (e.g., :has(div))
    arg_ident = MagicMock()
    arg_ident.value = "div"
    function_ident = MagicMock()
    function_ident.argument_types = ['IDENT']
    function_ident.arguments = [arg_ident]
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, function_ident)
    
    # Verify that add_post_condition was called for the IDENT type
    xpath_expr.add_post_condition.assert_called_with('descendant::div')
    assert result == xpath_expr

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    arg_number = MagicMock()
    arg_number.value = "123"
    function_number = MagicMock()
    function_number.argument_types = ['NUMBER']
    function_number.arguments = [arg_number]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, function_number)
    
    assert "Expected a single string or ident" in str(excinfo.value)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object and its arguments
    # We need to simulate a successful case with a NUMBER type
    mock_arg = MagicMock()
    mock_arg.value = '0'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]

    # Test successful execution: :eq(0) should result in position() = 1
    result = translator.xpath_eq_function(xpath, mock_function)
    assert result == xpath
    assert 'position() = 1' in str(xpath)

    # Test with a different index: :eq(2) should result in position() = 3
    mock_arg.value = '2'
    xpath_new = XPathExpr(path='p')
    translator.xpath_eq_function(xpath_new, mock_function)
    assert 'position() = 3' in str(xpath_new)

    # Test Error case: passing non-NUMBER type (e.g., STRING)
    mock_function.argument_types.return_value = ['STRING']
    mock_arg.value = '"not_a_number"'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_function)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object passed to the function
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input (NUMBER type)
    # Create a mock function with arguments containing a numeric value
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    arg_val = MagicMock()
    arg_val.value = '0'
    mock_function_valid.arguments = [arg_val]
    
    result = translator.xpath_gt_function(mock_xpath, mock_function_valid)
    
    # Verify that position() > 1 was added (since index 0 + 1 = 1)
    mock_xpath.add_post_condition.assert_called_with('position() > 1')
    assert result == mock_xpath

    # Case 2: Invalid input type (e.g., STRING instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [arg_val]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(mock_xpath, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Case 3: Another valid input (different number)
    mock_function_val2 = MagicMock()
    mock_function_val2.argument_types.return_value = ['NUMBER']
    arg_val2 = MagicMock()
    arg_val2.value = '5'
    mock_function_val2.arguments = [arg_val2]
    
    translator.xpath_gt_function(mock_xpath, mock_function_val2)
    # Verify it handles the index offset correctly (5 + 1 = 6)
    mock_xpath.add_post_condition.assert_called_with('position() > 6')
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mock for the XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument
    mock_func_string = MagicMock()
    mock_func_string.argument_types = ['STRING']
    mock_func_string.arguments = [MagicMock()]
    mock_func_string.arguments[0].value = '.baz'
    
    # Mock css_to_xpath behavior: translating '.baz' with prefix 'descendant::' to './/descendant::*[@class="baz"]' 
    # or similar depending on implementation, but here we just need it to return a string.
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="baz"]')
    
    result = translator.xpath_has_function(mock_xpath, mock_func_string)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::*[@class="baz"]')

    # Case 2: Valid IDENT argument
    mock_func_ident = MagicMock()
    mock_func_ident.argument_types = ['IDENT']
    mock_func_ident.arguments = [MagicMock()]
    mock_func_ident.arguments[0].value = 'div'
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(mock_xpath, mock_func_ident)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types = ['NUMBER']
    mock_func_invalid.arguments = [MagicMock()]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single string or ident for :has()" in str(excinfo.value)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mock function object for :has(".bar")
    mock_function = MagicMock()
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '.bar'
    
    # Mock css_to_xpath to return the expected descendant xpath
    translator.css_to_xpath = MagicMock(return_value='descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar") and (not(@class) or substring-before(substring-after(concat(" ", @class, " "), " "), " ") = "bar" or substring-after(concat(" ", @class, " "), " ") = "bar")]')
    # Since the real css_to_xpath is complex, we'll simplify the mock for testing logic
    translator.css_to_xpath = MagicMock(return_value='descendant::*.bar')

    # Test 1: Valid string argument
    result = translator.xpath_has_function(xpath_expr, mock_function)
    assert result == xpath_expr
    assert 'descendant::*.bar' in str(xpath_expr)

    # Test 2: Valid IDENT argument (e.g., :has(div))
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'div'
    translator.css_to_xpath.return_value = 'descendant::div'
    
    new_expr = XPathExpr(path='div', element='*')
    result = translator.xpath_has_function(new_expr, mock_function)
    assert 'descendant::div' in str(new_expr)

    # Test 3: Invalid argument type (e.g., NUMBER)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(XPathExpr(), mock_function)

    # Test 4: Invalid argument type (e.g., list of types not supported)
    mock_function.argument_types = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input (integer argument)
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '2'
    mock_function_valid.arguments = [mock_arg_valid]
    
    result = translator.xpath_eq_function(xpath_expr, mock_function_valid)
    
    assert result == xpath_expr
    # XPath position is 1-indexed, so index 2 becomes position() = 3
    xpath_expr.add_post_condition.assert_called_with('position() = 3')
    
    # Case 2: Invalid input type (e.g., STRING)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='not_a_number')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Case 3: Zero index (first element)
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero.arguments = [mock_arg_zero]
    
    translator.xpath_eq_function(xpath_expr, mock_function_zero)
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object and its arguments
    mock_function = MagicMock()
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    mock_function.arguments = [mock_arg]
    
    # Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    
    # We need to mock css_to_xpath because it's inherited from HTMLTranslator
    # and depends on the internal state of the translator/cssselect
    translator.css_to_xpath = MagicMock(return_value='descendant::*.bar')
    
    result = translator.xpath_has_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    assert 'descendant::*.bar' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    xpath_expr_2 = XPathExpr(path='div')
    result_2 = translator.xpath_has_function(xpath_expr_2, mock_function)
    assert 'descendant::*.bar' in str(xpath_expr_2)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    xpath_expr_3 = XPathExpr(path='div')
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr_3, mock_function)

    # Case 4: Testing with different selector content
    mock_function.argument_types.return_value = ['STRING']
    mock_arg.value = 'span'
    translator.css_to_xpath.return_value = 'descendant::span'
    
    xpath_expr_4 = XPathExpr(path='div')
    translator.xpath_has_function(xpath_expr_4, mock_function)
    assert 'descendant::span' in str(xpath_expr_4)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mocking the function argument object
    mock_function = MagicMock()
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '.bar'
    mock_function.argument_types = ['STRING']

    # Mock css_to_xpath to return a specific xpath string
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')

    # Execute the method
    result = translator.xpath_has_function(xpath_expr, mock_function)

    # Assertions
    assert result == xpath_expr
    assert 'descendant::*[@class="bar"]' in str(xpath_expr)

    # Test with invalid argument types (should raise ExpressionError)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr, mock_function)

    # Test with IDENT type
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'div'
    translator.css_to_xpath.return_value = 'descendant::div'
    result_ident = translator.xpath_has_function(xpath_expr, mock_function)
    assert 'descendant::div' in str(result_ident)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument (e.g., :has(".baz"))
    arg_string = MagicMock()
    arg_string.value = '.baz'
    function_string = MagicMock()
    function_string.argument_types = ['STRING']
    function_string.arguments = [arg_string]
    
    # Mocking css_to_xpath behavior for the test
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="baz"]')
    
    result = translator.xpath_has_function(xpath_expr, function_string)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::*[@class="baz"]')
    
    # Case 2: Valid IDENT argument (e.g., :has(div))
    arg_ident = MagicMock()
    arg_ident.value = 'div'
    function_ident = MagicMock()
    function_ident.argument_types = ['IDENT']
    function_ident.arguments = [arg_ident]
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, function_ident)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER)
    arg_number = MagicMock()
    arg_number.value = '1'
    function_error = MagicMock()
    function_error.argument_types = ['NUMBER']
    function_error.arguments = [arg_number]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, function_error)
    
    assert "Expected a single string or ident" in str(excinfo.value)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_arg = MagicMock()
    mock_arg.value = '1'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    # Test successful execution: :gt(1) should result in position() > 2
    result = translator.xpath_gt_function(xpath, mock_function)
    assert result == xpath
    assert 'position() > 2' in str(xpath)

    # Mock for an invalid argument type (e.g., STRING)
    mock_arg_invalid = MagicMock()
    mock_arg_invalid.value = 'foo'
    
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_invalid]
    
    # Test that ExpressionError is raised for non-NUMBER types
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test with index 0: :gt(0) should result in position() > 1
    xpath_zero = XPathExpr(path='div')
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_function_zero.arguments = [MagicMock(value='0')]
    
    translator.xpath_gt_function(xpath_zero, mock_function_zero)
    assert 'position() > 1' in str(xpath_zero)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    # Mocking the function object passed to the method
    mock_function = MagicMock()
    
    # Test case 1: Valid NUMBER argument (e.g., :lt(1) should result in position() < 2)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '1'
    
    result = translator.xpath_lt_function(xpath_expr, mock_function)
    assert result == xpath_expr
    assert 'position() < 2' in str(xpath_expr)
    
    # Test case 2: Valid NUMBER argument (e.g., :lt(0) should result in position() < 1)
    xpath_expr = XPathExpr(path='p')
    mock_function.arguments[0].value = '0'
    translator.xpath_lt_function(xpath_expr, mock_function)
    assert 'position() < 1' in str(xpath_expr)

    # Test case 3: Invalid argument type (e.g., STRING instead of NUMBER)
    xpath_expr = XPathExpr(path='p')
    mock_function.argument_types.return_value = ['STRING']
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value) or \
           "Expected a single integer for :lt()" in str(excinfo.value)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock for the function object passed to the method
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = "'test'"
    
    # We need to mock xpath_literal because it's called within the method
    translator.xpath_literal = MagicMock(return_value="'test'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    assert 'contains(., \'test\')' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments[0].value = 'some_id'
    translator.xpath_literal = MagicMock(return_value="'some_id'")
    
    # Reset xpath_expr for new test
    new_xpath_expr = XPathExpr(path='div')
    result = translator.xpath_contains_function(new_xpath_expr, mock_function)
    assert "contains(., 'some_id')" in str(new_xpath_expr)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., list of types)
    mock_function.argument_types.return_value = ['STRING', 'NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function argument object from cssselect
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'test'
    
    # We need to mock xpath_literal because it's inherited from HTMLTranslator
    translator.xpath_literal = MagicMock(return_value="'test'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    # Check if post_condition was added correctly
    assert 'contains(., \'test\')' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'some_id'
    translator.xpath_literal = MagicMock(return_value='some_id')
    
    # Reset xpath_expr for clean state in second check
    new_xpath = XPathExpr(path='div')
    translator.xpath_contains_function(new_xpath, mock_function)
    assert 'contains(., some_id)' in str(new_xpath)

    # Case 3: Invalid argument type (e.g., NUMBER)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Case 4: Another invalid argument type (e.g., Boolean/None)
    mock_function.argument_types = []
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function argument object passed by cssselect
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    
    mock_function = MagicMock()
    mock_function.arguments = [mock_arg]
    mock_function.argument_types = ['STRING']

    # Test case 1: Valid STRING argument
    # Should result in a post-condition using descendant axis
    translator.xpath_has_function(xpath_expr, mock_function)
    assert 'descendant::.bar' in str(xpath_expr)

    # Test case 2: Valid IDENT argument
    mock_arg.value = 'div'
    mock_function.argument_types = ['IDENT']
    xpath_expr_ident = XPathExpr(path='div')
    translator.xpath_has_function(xpath_expr_ident, mock_function)
    assert 'descendant::div' in str(xpath_expr_ident)

    # Test case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types = ['NUMBER']
    xpath_expr_error = XPathExpr(path='div')
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr_error, mock_function)

    # Test case 4: Verify it uses css_to_xpath via the logic (integration check)
    # Since we can't easily mock css_to_xpath without affecting class scope, 
    # we rely on the fact that 'descendant::' is prepended to the transformed value.
    # If input is '.foo', output should contain 'descendant::.foo' (assuming no complex CSS)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mocking XPathExpr to capture post-conditions
    xpath_expr = MagicMock(spec=XPathExpr)
    xpath_expr.add_post_condition = MagicMock()

    # Case 1: Valid input (NUMBER type)
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '0'
    mock_function_valid.arguments = [mock_arg_valid]

    result = translator.xpath_gt_function(xpath_expr, mock_function_valid)
    
    assert result == xpath_expr
    # :gt(0) should result in position() > 1 (since it's 0-indexed in jQuery but 1-indexed in XPath)
    xpath_expr.add_post_condition.assert_called_with('position() > 1')

    # Case 2: Valid input with different index
    mock_arg_other = MagicMock()
    mock_arg_other.value = '5'
    mock_function_valid.arguments = [mock_arg_other]
    
    translator.xpath_gt_function(xpath_expr, mock_function_valid)
    xpath_expr.add_post_condition.assert_called_with('position() > 6')

    # Case 3: Invalid input type (e.g., STRING instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='abc')]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument: :eq(0)
    mock_arg_0 = MagicMock()
    mock_arg_0.value = '0'
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_function_valid.arguments = [mock_arg_0]
    
    # Test valid case: :eq(0) should result in position() = 1
    result = translator.xpath_eq_function(xpath_expr, mock_function_valid)
    assert result == xpath_expr
    assert 'position() = 1' in str(xpath_expr)

    # Test valid case: :eq(2) should result in position() = 3
    xpath_expr_2 = XPathExpr(path='p')
    mock_arg_2 = MagicMock()
    mock_arg_2.value = '2'
    mock_function_valid_2 = MagicMock()
    mock_function_valid_2.argument_types.return_value = ['NUMBER']
    mock_function_valid_2.arguments = [mock_arg_2]
    
    translator.xpath_eq_function(xpath_expr_2, mock_function_valid_2)
    assert 'position() = 3' in str(xpath_expr_2)

    # Test invalid case: Non-NUMBER argument type should raise ExpressionError
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_0]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_invalid)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Test invalid case: Argument is not a number (simulated by argument type check)
    mock_function_not_num = MagicMock()
    mock_function_not_num.argument_types.return_value = ['IDENT']
    mock_function_not_num.arguments = [mock_arg_0]
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath_expr, mock_function_not_num)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock the function object and its arguments
    mock_function = MagicMock()
    mock_argument = MagicMock()
    mock_argument.value = 'title'
    mock_function.arguments = [mock_argument]
    
    # Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    # Mocking xpath_literal which is inherited from HTMLTranslator
    translator.xpath_literal = MagicMock(return_value="'title'")
    
    result = translator.xpath_contains_function(xpath, mock_function)
    
    assert result == xpath
    assert "contains(., 'title')" in str(xpath)

    # Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    xpath_new = XPathExpr(path='div')
    result_ident = translator.xpath_contains_function(xpath_new, mock_function)
    
    assert result_ident == xpath_new
    assert "contains(., 'title')" in str(xpath_new)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)

    # Case 4: Invalid argument type (e.g., BOOLEAN) should raise ExpressionError
    mock_function.argument_types.return_value = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)
```


