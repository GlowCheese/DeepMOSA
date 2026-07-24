####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object returned by cssselect parser
    mock_function = MagicMock()
    
    # Test Case 1: Valid NUMBER argument
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '0'
    
    translator.xpath_gt_function(xpath, mock_function)
    # :gt(0) should result in position() > 1
    assert 'position() > 1' in str(xpath)

    # Test Case 2: Valid NUMBER argument (different value)
    xpath_two = XPathExpr(path='p')
    mock_function.arguments[0].value = '2'
    translator.xpath_gt_function(xpath_two, mock_function)
    assert 'position() > 3' in str(xpath_two)

    # Test Case 3: Invalid argument type (STRING instead of NUMBER)
    xpath_error = XPathExpr(path='span')
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '"text"'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_error, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test Case 4: Invalid argument type (IDENT instead of NUMBER)
    mock_function.argument_types.return_value = ['IDENT']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_error, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div', element='*')
    
    # Mock the function argument object
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.arguments = [MagicMock(value='.bar')]
    mock_func.argument_types.return_value = ['STRING']
    
    # We need to mock css_to_xpath because it's a method of the translator
    # and relies on complex CSS parsing logic
    translator.css_to_xpath = MagicMock(return_value='descendant::*.bar')
    
    result = translator.xpath_has_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert 'descendant::*.bar' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_func.arguments = [MagicMock(value='div')]
    mock_func.argument_types.return_value = ['IDENT']
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    xpath_expr_2 = XPathExpr(path='*', element='*')
    translator.xpath_has_function(xpath_expr_2, mock_func)
    assert 'descendant::div' in str(xpath_expr_2)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func.arguments = [MagicMock(value='123')]
    mock_func.argument_types.return_value = ['NUMBER']
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident for :has(), got" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., BOOLEAN) should raise ExpressionError
    mock_func.arguments = [MagicMock(value='true')]
    mock_func.argument_types.return_value = ['BOOLEAN']
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident for :has(), got" in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # 1. Test valid STRING argument
    mock_arg_string = MagicMock()
    mock_arg_string.value = "test-text"
    
    mock_function_string = MagicMock()
    mock_function_string.argument_types = ['STRING']
    mock_function_string.arguments = [mock_arg_string]
    
    # Mocking cssselect_xpath.xpath_literal behavior 
    # (Assuming it wraps the string in quotes for XPath)
    translator.xpath_literal = MagicMock(return_value="'test-text'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function_string)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with("contains(., 'test-text')")

    # 2. Test valid IDENT argument
    mock_arg_ident = MagicMock()
    mock_arg_ident.value = some_id
    # Note: In a real test, 'some_id' would be a string variable
    # but for the mock we just need the structure
    
    mock_function_ident = MagicMock()
    mock_function_ident.argument_types = ['IDENT']
    mock_function_ident.arguments = [mock_arg_ident]
    
    translator.xpath_literal = MagicMock(return_value="some_id")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function_ident)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with("contains(., some_id)")

    # 3. Test invalid argument type (e.g., NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types = ['NUMBER']
    mock_function_invalid.arguments = [mock_arg_string]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # 4. Test invalid argument type (e.g., BOOLEAN)
    mock_function_bool = MagicMock()
    mock_function_bool.argument_types = ['BOOLEAN']
    mock_function_bool.arguments = [mock_arg_string]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr, mock_function_bool)
        
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Test Case 1: Successful execution with valid integer argument
    mock_arg = MagicMock()
    mock_arg.value = '0'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    result = translator.xpath_eq_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
    
    # Test Case 2: Successful execution with another integer argument
    mock_arg_1 = MagicMock()
    mock_arg_1.value = '5'
    
    mock_function_1 = MagicMock()
    mock_function_1.argument_types.return_value = ['NUMBER']
    mock_function_1.arguments = [mock_arg_1]
    
    translator.xpath_eq_function(xpath_expr, mock_function_1)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')

    # Test Case 3: Failure with invalid argument type (e.g., STRING)
    mock_arg_invalid = MagicMock()
    mock_arg_invalid.value = 'not_a_number'
    
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_invalid]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Test Case 4: Failure with empty arguments
    mock_function_empty = MagicMock()
    mock_function_empty.argument_types.return_value = ['NUMBER']
    mock_function_empty.arguments = []
    
    with pytest.raises(IndexError):
        translator.xpath_eq_function(xpath_expr, mock_function_empty)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    # Mocking XPathExpr instance
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # 1. Test valid numeric input (index 0)
    mock_arg_0 = MagicMock()
    mock_arg_0.value = '0'
    mock_func_0 = MagicMock()
    mock_func_0.argument_types.return_value = ['NUMBER']
    mock_func_0.arguments = [mock_arg_0]
    
    translator.xpath_eq_function(xpath_expr, mock_func_0)
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
    
    # Reset mock for next scenario
    xpath_expr.add_post_condition.reset_mock()

    # 2. Test valid numeric input (index 5)
    mock_arg_5 = MagicMock()
    mock_arg_5.value = '5'
    mock_func_5 = MagicMock()
    mock_func_5.argument_types.return_value = ['NUMBER']
    mock_func_5.arguments = [mock_arg_5]
    
    translator.xpath_eq_function(xpath_expr, mock_func_5)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')

    # 3. Test invalid argument type (STRING instead of NUMBER)
    mock_arg_str = MagicMock()
    mock_arg_str.value = 'not_a_number'
    mock_func_str = MagicMock()
    mock_func_str.argument_types.return_value = ['STRING']
    mock_func_str.arguments = [mock_arg_str]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_func_str)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object for :eq()
    # Case 1: Valid numeric argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '0'
    mock_func_valid.arguments = [mock_arg_valid]
    
    result = translator.xpath_eq_function(xpath_expr, mock_func_valid)
    assert result == xpath_expr
    # position() = 0 + 1 -> position() = 1
    assert 'position() = 1' in str(xpath_expr)

    # Case 2: Valid numeric argument (different index)
    mock_arg_index_2 = MagicMock()
    mock_arg_index_2.value = '2'
    mock_func_valid.arguments = [mock_arg_index_2]
    
    xpath_expr_new = XPathExpr(path='p')
    translator.xpath_eq_function(xpath_expr_new, mock_func_valid)
    assert 'position() = 3' in str(xpath_expr_new)

    # Case 3: Invalid argument type (e.g., STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='not_a_number')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, mock_func_invalid)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., IDENT)
    mock_func_ident = MagicMock()
    mock_func_ident.argument_types.return_value = ['IDENT']
    mock_func_ident.arguments = [MagicMock(value='foo')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath_expr, mock_func_ident)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    # Mocking XPathExpr (which inherits from XPathExprOrig)
    # We need to mock the behavior of add_post_condition and the structure
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function argument
    def create_mock_arg(value, arg_types):
        arg = MagicMock()
        arg.value = value
        return arg

    # Case 1: Valid integer input (0-indexed in jQuery, becomes position() = 1 in XPath)
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [create_mock_arg('0', ['NUMBER'])]
    
    translator.xpath_eq_function(mock_xpath, mock_func_valid)
    mock_xpath.add_post_condition.assert_called_with('position() = 1')

    # Case 2: Another valid integer input
    mock_func_valid_2 = MagicMock()
    mock_func_valid_2.argument_types.return_value = ['NUMBER']
    mock_func_valid_2.arguments = [create_mock_arg('5', ['NUMBER'])]
    
    translator.xpath_eq_function(mock_xpath, mock_func_valid_2)
    mock_xpath.add_post_condition.assert_called_with('position() = 6')

    # Case 3: Invalid argument type (e.g., STRING instead of NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [create_mock_arg('abc', ['STRING'])]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock function object for a valid numeric argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    arg_valid = MagicMock()
    arg_valid.value = '1'
    mock_func_valid.arguments = [arg_valid]

    # Test valid case: :gt(0) should result in position() > 1
    result_expr = translator.xpath_gt_function(xpath_expr, mock_func_valid)
    assert str(result_expr).endswith('[position() > 2]')
    assert result_expr == xpath_expr

    # Test invalid case: Non-numeric argument type should raise ExpressionError
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    arg_invalid = MagicMock()
    arg_invalid.value = 'abc'
    mock_func_invalid.arguments = [arg_invalid]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(XPathExpr(path='div'), mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test edge case: :gt(-1) should result in position() > 0 (effectively any element)
    mock_func_edge = MagicMock()
    mock_func_edge.argument_types.return_value = ['NUMBER']
    arg_edge = MagicMock()
    arg_edge.value = '-1'
    mock_func_edge.arguments = [arg_edge]
    
    expr_edge = XPathExpr(path='div')
    translator.xpath_gt_function(expr_edge, mock_func_edge)
    assert '[position() > 0]' in str(expr_edge)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mock for the XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Mock for the function argument (simulating CSS selector :eq(n))
    # We need to simulate an object that has .argument_types and .arguments[0].value
    class MockFunction:
        def __init__(self, arg_types, value):
            self.argument_types = arg_types
            self.arguments = [MagicMock()]
            self.arguments[0].value = value

    # Case 1: Valid input (Integer)
    func_valid = MockFunction(['NUMBER'], '0')
    translator.xpath_eq_function(mock_xpath, func_valid)
    # XPath position is 1-based, so :eq(0) should be position() = 1
    mock_xpath.add_post_condition.assert_called_with('position() = 1')

    # Case 2: Valid input (Higher index)
    func_valid_2 = MockFunction(['NUMBER'], '5')
    translator.xpath_eq_function(mock_xpath, func_valid_2)
    mock_xpath.add_post_condition.assert_called_with('position() = 6')

    # Case 3: Invalid input type (String instead of Number)
    func_invalid = MockFunction(['STRING'], '"text"')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, func_invalid)
    assert "Expected a single integer" in str(excinfo.value)

    # Case 4: Invalid input type (Ident instead of Number)
    func_invalid_2 = MockFunction(['IDENT'], 'some_id')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, func_invalid_2)
    assert "Expected a single integer" in str(excinfo.value)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Test Case 1: Valid integer argument
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '1'
    
    # Expected: position() > (1 + 1) -> position() > 2
    translator.xpath_gt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() > 2')
    
    # Reset mock for next test case
    xpath_expr.add_post_condition.reset_mock()

    # Test Case 2: Invalid argument type (e.g., STRING)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock()]
    mock_function_invalid.arguments[0].value = 'abc'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)
    xpath_expr.add_post_condition.assert_not_called()

    # Test Case 3: Zero index
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_function_zero.arguments = [MagicMock()]
    mock_function_zero.arguments[0].value = '0'
    
    translator.xpath_gt_function(xpath_expr, mock_function_zero)
    xpath_expr.add_post_condition.assert_called_with('position() > 1')
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock function object for :contains("text")
    mock_func = MagicMock()
    mock_func.argument_types = ['STRING']
    
    # Create a mock argument with the value 'target'
    mock_arg = MagicMock()
    mock_arg.value = 'target'
    mock_func.arguments = [mock_arg]
    
    # Mock xpath_literal to return a properly formatted XPath string literal
    translator.xpath_literal = MagicMock(return_value="'target'")

    # Execute the function
    result = translator.xpath_contains_function(xpath_expr, mock_func)

    # Assertions
    assert result == xpath_expr
    assert 'contains(., \'target\')' in str(xpath_expr)

    # Test case for invalid argument type (e.g., NUMBER instead of STRING)
    mock_func.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr, mock_func)

    # Test case for IDENT type (should be allowed)
    mock_func.argument_types = ['IDENT']
    mock_arg.value = 'some_ident'
    translator.xpath_literal.return_value = 'some_ident'
    
    new_xpath = XPathExpr(path='div')
    translator.xpath_contains_function(new_xpath, mock_func)
    assert 'contains(., some_ident)' in str(new_xpath)
```


# LLM-generated content at query #12
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
    
    # Test valid input: :gt(0) should result in position() > 1
    result_valid = translator.xpath_gt_function(xpath, mock_func_valid)
    assert result_valid == xpath
    assert 'position() > 1' in str(xpath)
    
    # Test valid input: :gt(2) should result in position() > 3
    mock_arg_val2 = MagicMock()
    mock_arg_val2.value = '2'
    mock_func_valid.arguments = [mock_arg_val2]
    xpath_new = XPathExpr(path='div')
    translator.xpath_gt_function(xpath_new, mock_func_valid)
    assert 'position() > 3' in str(xpath_new)

    # Mock for invalid argument type (e.g., STRING instead of NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='foo')]

    # Test invalid input: should raise ExpressionError
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid function argument (NUMBER)
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [MagicMock(value='1')]

    # Test successful execution: :gt(1) should result in position() > 2
    result = translator.xpath_gt_function(xpath, mock_func_valid)
    assert result == xpath
    assert 'position() > 2' in str(xpath)

    # Mock for an invalid function argument type (STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='"text"')]

    # Test that ExpressionError is raised for non-NUMBER types
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test with index 0: :gt(0) should result in position() > 1
    xpath_zero = XPathExpr(path='div')
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_func_zero.arguments = [MagicMock(value='0')]
    
    translator.xpath_gt_function(xpath_zero, mock_func_zero)
    assert 'position() > 1' in str(xpath_zero)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed by cssselect
    mock_function = MagicMock()
    
    # Test case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'hello'
    # We need to mock xpath_literal because it is called by xpath_contains_function
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    assert 'contains(., \'hello\')' in str(xpath_expr)

    # Test case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments[0].value = 'some_id'
    translator.xpath_literal = MagicMock(return_value='some_id')
    
    new_expr = XPathExpr(path='div')
    translator.xpath_contains_function(new_expr, mock_function)
    assert 'contains(., some_id)' in str(new_expr)

    # Test case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)

    # Test case 4: Invalid argument type (e.g., BOOLEAN) should raise ExpressionError
    mock_function.argument_types.return_value = ['BOOLEAN']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    # Mocking XPathExpr which is the xpathexpr_cls of the translator
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input (NUMBER type)
    # We need to mock the function object and its arguments structure
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    
    # Mocking the argument value access: function.arguments[0].value
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_func_valid.arguments = [mock_arg]
    
    translator.xpath_eq_function(mock_xpath, mock_func_valid)
    # position() = 0 + 1 -> position() = 1
    mock_xpath.add_post_condition.assert_called_with('position() = 1')

    # Case 2: Valid input with different index
    mock_arg.value = '5'
    translator.xpath_eq_function(mock_xpath, mock_func_valid)
    mock_xpath.add_post_condition.assert_called_with('position() = 6')

    # Case 3: Invalid input type (not NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [mock_arg]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed to the method
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'hello'
    
    # We need to mock xpath_literal because it is called within the method
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    # The post_condition should be contains(., 'hello')
    # Note: XPathExpr uses add_post_condition which appends to self.post_condition
    assert "contains(., 'hello')" in str(xpath_expr)

    # Case 2: Valid IDENT argument (e.g., a class name or unquoted string)
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments[0].value = 'title'
    translator.xpath_literal = MagicMock(return_value='title')
    
    # Reset xpath_expr for clean test
    new_xpath = XPathExpr(path='h1')
    translator.xpath_contains_function(new_xpath, mock_function)
    assert "contains(., title)" in str(new_xpath)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)

    # Case 4: Invalid argument type (e.g., LIST or other types)
    mock_function.argument_types.return_value = ['LIST']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object and its arguments
    mock_func = MagicMock()
    mock_arg = MagicMock()
    
    # Test case 1: Valid input (index 0)
    mock_func.argument_types.return_value = ['NUMBER']
    mock_arg.value = '0'
    mock_func.arguments = [mock_arg]
    
    translator.xpath_eq_function(xpath, mock_func)
    # position() = value + 1 -> position() = 1
    assert 'position() = 1' in str(xpath)

    # Test case 2: Valid input (index 5)
    mock_arg.value = '5'
    # Reset xpath for fresh check
    new_xpath = XPathExpr(path='p')
    translator.xpath_eq_function(new_xpath, mock_func)
    assert 'position() = 6' in str(new_xpath)

    # Test case 3: Invalid input type (STRING instead of NUMBER)
    mock_func.argument_types.return_value = ['STRING']
    mock_arg.value = '"not_a_number"'
    mock_func.arguments = [mock_arg]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_func)
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object passed to xpath_eq_function
    mock_function = MagicMock()
    
    # Case 1: Valid input (integer)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_function.arguments = [mock_arg]
    
    translator.xpath_eq_function(xpath, mock_function)
    # position() should be index + 1 (0 + 1 = 1)
    assert '[position() = 1]' in str(xpath)
    
    # Case 2: Valid input (different integer)
    mock_arg.value = '5'
    xpath_new = XPathExpr(path='p')
    translator.xpath_eq_function(xpath_new, mock_function)
    assert '[position() = 6]' in str(xpath_new)

    # Case 3: Invalid input type (e.g., STRING instead of NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    mock_arg.value = 'not_a_number'
    mock_function.arguments = [mock_arg]
    
    xpath_invalid = XPathExpr(path='span')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_invalid, mock_function)
    
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument (e.g., :has(".bar"))
    mock_arg_string = MagicMock()
    mock_arg_string.value = ".bar"
    
    function_string = MagicMock()
    function_string.argument_types = ['STRING']
    function_string.arguments = [mock_arg_string]
    
    # Mock css_to_xpath to return a valid xpath fragment
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')
    
    result = translator.xpath_has_function(xpath_expr, function_string)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::*[@class="bar"]')

    # Case 2: Valid IDENT argument (e.g., :has(div))
    mock_arg_ident = MagicMock()
    mock_arg_ident.value = "div"
    
    function_ident = MagicMock()
    function_ident.argument_types = ['IDENT']
    function_ident.arguments = [mock_arg_ident]
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(xpath_expr, function_ident)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER)
    mock_arg_number = MagicMock()
    mock_arg_number.value = "1"
    
    function_number = MagicMock()
    function_number.argument_types = ['NUMBER']
    function_number.arguments = [mock_arg_number]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, function_number)
    
    assert "Expected a single string or ident for :has()" in str(excinfo.value)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock function object for a valid numeric argument: :gt(0)
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [MagicMock(value='0')]
    
    # Test successful execution
    result = translator.xpath_gt_function(xpath_expr, mock_func_valid)
    assert result == xpath_expr
    assert 'position() > 1' in str(xpath_expr)

    # Mock function object for an invalid argument type: :gt("string")
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='not_a_number')]
    
    # Test that it raises ExpressionError for non-numeric types
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath_expr, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test with a different valid numeric argument: :gt(5)
    xpath_expr_two = XPathExpr(path='p')
    mock_func_two = MagicMock()
    mock_func_two.argument_types.return_value = ['NUMBER']
    mock_func_two.arguments = [MagicMock(value='5')]
    
    translator.xpath_gt_function(xpath_expr_two, mock_func_two)
    assert 'position() > 6' in str(xpath_expr_two)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object with argument types and values
    # Case 1: Valid NUMBER input (0-indexed in jQuery, converted to 1-indexed in XPath)
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [MagicMock(value='0')]
    
    translator.xpath_eq_function(xpath, mock_func_valid)
    assert 'position() = 1' in str(xpath)

    # Case 2: Valid NUMBER input (index 5)
    mock_func_5 = MagicMock()
    mock_func_5.argument_types.return_value = ['NUMBER']
    mock_func_5.arguments = [MagicMock(value='5')]
    
    translator.xpath_eq_function(xpath, mock_func_5)
    assert 'position() = 6' in str(xpath)

    # Case 3: Invalid argument type (STRING instead of NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [MagicMock(value='"0"')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)

    # Case 4: Invalid argument type (IDENT instead of NUMBER)
    mock_func_ident = MagicMock()
    mock_func_ident.argument_types.return_value = ['IDENT']
    mock_func_ident.arguments = [MagicMock(value='foo')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_func_ident)
    assert "Expected a single integer for :eq(), got" in str(excinfo.value)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')

    # Mocking the function object provided by cssselect parser
    # Case 1: Valid numeric input (e.g., :gt(0) should result in position() > 1)
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '0'
    mock_function_valid.arguments = [mock_arg_valid]

    result_xpath = translator.xpath_gt_function(xpath, mock_function_valid)
    assert result_xpath == xpath
    # In XPath, :gt(0) means index > 0 (0-based in jQuery), so position() > 1
    assert 'position() > 1' in str(xpath)

    # Case 2: Valid numeric input with different value (e.g., :gt(2))
    mock_function_val2 = MagicMock()
    mock_function_val2.argument_types.return_value = ['NUMBER']
    mock_arg_val2 = MagicMock()
    mock_arg_val2.value = '2'
    mock_function_val2.arguments = [mock_arg_val2]
    
    xpath_new = XPathExpr(path='p')
    translator.xpath_gt_function(xpath_new, mock_function_val2)
    assert 'position() > 3' in str(xpath_new)

    # Case 3: Invalid argument type (e.g., passing a string instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='abc')]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object and its arguments
    # We need to mock an argument that has a .value attribute
    mock_arg = MagicMock()
    mock_arg.value = '0'
    
    mock_function = MagicMock()
    mock_function.arguments = [mock_arg]
    mock_function.argument_types.return_value = ['NUMBER']

    # Test successful execution: :gt(0) should result in position() > 1
    result_xpath = translator.xpath_gt_function(xpath, mock_function)
    assert result_xpath == xpath
    assert 'position() > 1' in str(xpath)

    # Test with different number: :gt(2) should result in position() > 3
    mock_arg.value = '2'
    xpath_new = XPathExpr(path='p')
    translator.xpath_gt_function(xpath_new, mock_function)
    assert 'position() > 3' in str(xpath_new)

    # Test error case: Incorrect argument type (e.g., STRING instead of NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(XPathExpr(), mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test error case: Passing non-numeric value string (though cssselect usually handles parsing, 
    # the implementation uses int() which will raise ValueError if it's not a digit string)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_arg.value = 'abc'
    with pytest.raises(ValueError):
        translator.xpath_gt_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object that would be passed to the function
    mock_xpath = MagicMock()
    
    # Create a mock for the function argument
    # We need to simulate an argument with value and type
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.arguments = [MagicMock(value='.bar')]
    mock_func.argument_types.return_value = ['STRING']
    
    # We need to mock css_to_xpath because it's a method of the translator
    # In a real scenario, '.bar' becomes 'descendant::*[@class="bar"]' or similar
    translator.css_to_xpath = MagicMock(return_value='descendant::*[@class="bar"]')
    
    result = translator.xpath_has_function(mock_xpath, mock_func)
    
    # Assertions for Case 1
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::*[@class="bar"]')

    # Case 2: Valid IDENT argument
    mock_func.arguments = [MagicMock(value='div')]
    mock_func.argument_types.return_value = ['IDENT']
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(mock_xpath, mock_func)
    
    # Assertions for Case 2
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func.arguments = [MagicMock(value='123')]
    mock_func.argument_types.return_value = ['NUMBER']
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(mock_xpath, mock_func)
    
    assert "Expected a single string or ident for :has(), got" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., BOOLEAN) should raise ExpressionError
    mock_func.arguments = [MagicMock(value='true')]
    mock_func.argument_types.return_value = ['BOOLEAN']
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(mock_xpath, mock_func)
    
    assert "Expected a single string or ident for :has(), got" in str(excinfo.value)
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object provided by cssselect parser
    mock_function = MagicMock()
    
    # Case 1: Valid NUMBER argument (e.g., :gt(0) should result in position() > 1)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_function.arguments = [mock_arg]
    
    result_xpath = translator.xpath_gt_function(xpath, mock_function)
    assert result_xpath == xpath
    assert 'position() > 1' in str(xpath)

    # Case 2: Valid NUMBER argument (e.g., :gt(2) should result in position() > 3)
    xpath_new = XPathExpr(path='div')
    mock_arg_two = MagicMock()
    mock_arg_two.value = '2'
    mock_function.arguments = [mock_arg_two]
    
    translator.xpath_gt_function(xpath_new, mock_function)
    assert 'position() > 3' in str(xpath_new)

    # Case 3: Invalid argument type (e.g., STRING instead of NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    mock_arg_string = MagicMock()
    mock_arg_string.value = 'abc'
    mock_function.arguments = [mock_arg_string]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function argument structure expected by the method
    # function.arguments[0].value should be the selector string
    mock_func = MagicMock()
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    mock_func.arguments = [mock_arg]
    # The method checks for ['STRING'] or ['IDENT']
    mock_func.argument_types.return_value = ['STRING']

    # Mocking css_to_xpath to return a specific xpath string
    translator.css_to_xpath = MagicMock(return_value='descendant::*.bar')

    # Execute the method
    result = translator.xpath_has_function(xpath_expr, mock_func)

    # Assertions
    assert result == xpath_expr
    # Check if post_condition was added correctly via the mocked css_to_xpath output
    assert 'descendant::*.bar' in str(xpath_expr)

    # Test with ExpressionError for invalid argument types
    mock_func.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_expr, mock_func)

    # Test with IDENT type (e.g., unquoted tag name)
    mock_func.argument_types.return_value = ['IDENT']
    mock_arg.value = 'div'
    translator.css_to_xpath.return_value = 'descendant::div'
    
    result_ident = translator.xpath_has_function(xpath_expr, mock_func)
    assert 'descendant::div' in str(result_ident)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_first_pseudo():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    result = translator.xpath_first_pseudo(xpath_expr)
    
    assert result is xpath_expr
    assert xpath_expr.post_condition == 'position() = 1'
    assert str(xpath_expr) == "p[position() = 1]"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_image_pseudo():
    # Arrange
    translator = JQueryTranslator()
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Act
    result = translator.xpath_image_pseudo(mock_xpath)
    
    # Assert
    # The method should call add_condition with the specific XPath for type='image'
    mock_xpath.add_condition.assert_called_once_with("@type = 'image' and name(.) = 'input'")
    # The method should return the xpath object itself to allow chaining
    assert result is mock_xpath
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object provided by cssselect parser
    mock_function = MagicMock()
    
    # Case 1: Valid input (NUMBER type)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '0'
    
    translator.xpath_gt_function(xpath, mock_function)
    assert 'position() > 1' in str(xpath)

    # Case 2: Valid input with different number
    xpath_new = XPathExpr(path='p')
    mock_function.arguments[0].value = '5'
    translator.xpath_gt_function(xpath_new, mock_function)
    assert 'position() > 6' in str(xpath_new)

    # Case 3: Invalid input type (e.g., STRING instead of NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = 'abc'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Case 4: Testing the logic of index offset (gt(0) should match position > 1)
    # In XPath, position() is 1-based. jQuery :gt(0) means indices > 0, 
    # which corresponds to elements from index 1 onwards in 0-based, or position > 1.
    xpath_zero = XPathExpr(path='span')
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments[0].value = '0'
    translator.xpath_gt_function(xpath_zero, mock_function)
    assert "position() > 1" in str(xpath_zero)
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
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    # Mock XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # 1. Test valid input: :lt(1) should result in position() < 2
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [MagicMock(value='1')]
    
    translator.xpath_lt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() < 2')

    # 2. Test valid input: :lt(0) should result in position() < 1
    mock_function.arguments = [MagicMock(value='0')]
    translator.xpath_lt_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('position() < 1')

    # 3. Test invalid input type: Should raise ExpressionError if not 'NUMBER'
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock(value='"text"')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function)
    assert "Expected a single integer for :gt()" in str(excinfo.value) or "gt" in str(excinfo.value) 
    # Note: The original code contains a copy-paste error where it references :gt() 
    # inside the xpath_lt_function error message, so we test for that specific behavior.
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mocking XPathExpr (the xpath object passed to the function)
    xpath_mock = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid input - integer argument
    # Create a mock for the function argument with value '1'
    mock_function = MagicMock()
    mock_argument = MagicMock()
    mock_argument.value = '1'
    mock_function.arguments = [mock_argument]
    mock_function.argument_types = ['NUMBER']
    
    translator.xpath_lt_function(xpath_mock, mock_function)
    # For :lt(1), xpath position should be < (1 + 1) -> 'position() < 2'
    xpath_mock.add_post_condition.assert_called_with('position() < 2')
    
    # Case 2: Invalid input - non-numeric argument type
    # Reset mock for next assertion
    xpath_mock.reset_mock()
    
    mock_function_invalid = MagicMock()
    mock_function_invalid.arguments = [mock_argument]
    mock_function_invalid.argument_types = ['STRING'] # Not 'NUMBER'
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_mock, mock_function_invalid)
    
    assert "Expected a single integer" in str(excinfo.value)
    xpath_mock.add_post_condition.assert_not_called()

    # Case 3: Valid input - zero index
    xpath_mock.reset_mock()
    mock_function_zero = MagicMock()
    mock_argument_zero = MagicMock()
    mock_argument_zero.value = '0'
    mock_function_zero.arguments = [mock_argument_zero]
    mock_function_zero.argument_types = ['NUMBER']
    
    translator.xpath_lt_function(xpath_mock, mock_function_zero)
    # For :lt(0), position should be < (0 + 1) -> 'position() < 1'
    xpath_mock.add_post_condition.assert_called_with('position() < 1')
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock the function object passed by cssselect
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.argument_types.return_value = ['STRING']
    mock_func.arguments = [MagicMock(value='"target text"')]
    
    # Mock xpath_literal to return the value as is for simplicity in test
    translator.xpath_literal = MagicMock(side_effect=lambda x: x)
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    assert result == xpath_expr
    # Check if post_condition was added correctly
    assert 'contains(., "target text")' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_func.argument_types.return_value = ['IDENT']
    mock_func.arguments = [MagicMock(value='some_id')]
    
    xpath_expr_2 = XPathExpr(path='div')
    translator.xpath_contains_function(xpath_expr_2, mock_func)
    assert 'contains(., some_id)' in str(xpath_expr_2)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func.argument_types.return_value = ['NUMBER']
    mock_func.arguments = [MagicMock(value='123')]
    
    xpath_expr_3 = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_3, mock_func)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., BOOLEAN)
    mock_func.argument_types.return_value = ['BOOLEAN']
    xpath_expr_4 = XPathExpr(path='div')
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr_4, mock_func)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock function object with arguments
    mock_function = MagicMock()
    
    # Test Case 1: Valid STRING argument
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock(value='hello')]
    # We mock xpath_literal to return a formatted string for the XPath
    translator.xpath_literal = MagicMock(return_value="'hello'")
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    # Check if post condition was added correctly: contains(., 'hello')
    assert 'contains(., \'hello\')' in str(xpath_expr)

    # Test Case 2: Valid IDENT argument
    mock_function.argument_types = ['IDENT']
    mock_function.arguments = [MagicMock(value='title')]
    translator.xpath_literal = MagicMock(return_value='title')
    
    new_expr = XPathExpr(path='div')
    translator.xpath_contains_function(new_expr, mock_function)
    assert 'contains(., title)' in str(new_expr)

    # Test Case 3: Invalid argument type (NUMBER) should raise ExpressionError
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [MagicMock(value=123)]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Test Case 4: Invalid argument type (LIST/MISMATCH)
    mock_function.argument_types = ['LIST']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(path='div'), mock_function)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr (or any object with add_post_condition)
    mock_xpath = MagicMock()
    
    # Case 1: Valid NUMBER argument (e.g., :eq(0))
    # We need to mock a function object that has argument_types and arguments attributes
    mock_function_valid = MagicMock()
    mock_function_valid.argument_types.return_value = ['NUMBER']
    
    # Mocking the argument value (0)
    mock_arg = MagicMock()
    mock_arg.value = '0'
    mock_function_valid.arguments = [mock_arg]
    
    result = translator.xpath_eq_function(mock_xpath, mock_function_valid)
    
    # Assertions for valid case
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('position() = 1')
    
    # Reset mock for next test case
    mock_xpath.reset_mock()

    # Case 2: Invalid argument type (e.g., STRING instead of NUMBER)
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(mock_xpath, mock_function_invalid)
    
    assert "Expected a single integer for :eq()" in str(excinfo.value)
    mock_xpath.add_post_condition.assert_not_called()

    # Case 3: Valid NUMBER argument with different index (e.g., :eq(5))
    mock_function_index_5 = MagicMock()
    mock_function_index_5.argument_types.return_value = ['NUMBER']
    mock_arg_5 = MagicMock()
    mock_arg_5.value = '5'
    mock_function_index_5.arguments = [mock_arg_5]

    translator.xpath_eq_function(mock_xpath, mock_function_index_5)
    mock_xpath.add_post_condition.assert_called_with('position() = 6')
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object passed to xpath_gt_function
    mock_func = MagicMock()
    mock_func.argument_types.return_value = ['NUMBER']
    
    # Create a mock argument with a value
    mock_arg = MagicMock()
    mock_arg.value = '1'
    mock_func.arguments = [mock_arg]

    # Test successful execution: :gt(1) should result in position() > 2 (since it is 0-indexed in jQuery)
    result_xpath = translator.xpath_gt_function(xpath, mock_func)
    assert result_xpath == xpath
    assert 'position() > 2' in str(xpath)

    # Test error handling for incorrect argument types (e.g., STRING instead of NUMBER)
    mock_func.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_func)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test error handling when arguments are empty or invalid format
    mock_func.argument_types.return_value = ['NUMBER']
    mock_func.arguments = [] # No arguments provided
    with pytest.raises(IndexError):
        translator.xpath_gt_function(xpath, mock_func)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function argument
    def create_mock_arg(value, arg_types):
        arg = MagicMock()
        arg.value = value
        return arg

    # 1. Test successful execution for :lt(1) -> position() < 2
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_func_valid.arguments = [create_mock_arg('1', 'NUMBER')]
    
    translator.xpath_lt_function(mock_xpath, mock_func_valid)
    mock_xpath.add_post_condition.assert_called_once_with('position() < 2')
    
    # Reset mock for next scenario
    mock_xpath.add_post_condition.reset_mock()

    # 2. Test successful execution for :lt(0) -> position() < 1
    mock_func_zero = MagicMock()
    mock_func_zero.argument_types.return_value = ['NUMBER']
    mock_func_zero.arguments = [create_mock_arg('0', 'NUMBER')]
    
    translator.xpath_lt_function(mock_xpath, mock_func_zero)
    mock_xpath.add_post_condition.assert_called_once_with('position() < 1')

    # 3. Test error when argument type is not NUMBER (e.g., STRING)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_func_invalid.arguments = [create_mock_arg('abc', 'STRING')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(mock_xpath, mock_func_invalid)
    
    assert "Expected a single integer for :gt(), got" in str(excinfo.value)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='p')
    
    # Mocking the function object returned by cssselect parser
    mock_function = MagicMock()
    
    # Case 1: Valid input (NUMBER) - :lt(1) should result in position() < 2
    mock_function.argument_types.return_value = ['NUMBER']
    mock_arg = MagicMock()
    mock_arg.value = '1'
    mock_function.arguments = [mock_arg]
    
    translator.xpath_lt_function(xpath, mock_function)
    assert xpath.post_condition == 'position() < 2'
    
    # Case 2: Invalid input type (STRING instead of NUMBER)
    xpath_invalid = XPathExpr(path='p')
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='foo')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_invalid, mock_function_invalid)
    assert "Expected a single integer for :gt(), got" in str(excinfo.value)

    # Case 3: Testing with zero index - :lt(0) should result in position() < 1
    xpath_zero = XPathExpr(path='p')
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero.arguments = [mock_arg_zero]
    
    translator.xpath_lt_function(xpath_zero, mock_function_zero)
    assert xpath_zero.post_condition == 'position() < 1'
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid STRING argument
    mock_func_string = MagicMock()
    mock_func_string.argument_types = ['STRING']
    mock_func_string.arguments = [MagicMock()]
    mock_func_string.arguments[0].value = 'title'
    
    # Mocking cssselect_xpath.XPathExpr.xpath_literal behavior 
    # (Since we can't easily mock the base class method without changing globals, 
    # we assume it returns a quoted string for the test)
    translator.xpath_literal = MagicMock(return_value="'title'")
    
    result = translator.xpath_contains_function(mock_xpath, mock_func_string)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with("contains(., 'title')")

    # Case 2: Valid IDENT argument
    mock_func_ident = MagicMock()
    mock_func_ident.argument_types = ['IDENT']
    mock_func_ident.arguments = [MagicMock()]
    mock_func_ident.arguments[0].value = 'some_id'
    translator.xpath_literal = MagicMock(return_value="some_id")

    result = translator.xpath_contains_function(mock_xpath, mock_func_ident)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with("contains(., some_id)")

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types = ['NUMBER']
    mock_func_invalid.arguments = [MagicMock()]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath, mock_func_invalid)

    # Case 4: Invalid argument type (e.g., LIST/BOOLEAN)
    mock_func_unsupported = MagicMock()
    mock_func_unsupported.argument_types = ['BOOLEAN']
    mock_func_unsupported.arguments = [MagicMock()]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath, mock_func_unsupported)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    # Mock the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Test case 1: Valid numeric argument (index 0)
    mock_arg_0 = MagicMock()
    mock_arg_0.value = '0'
    function_0 = MagicMock()
    function_0.argument_types.return_value = ['NUMBER']
    function_0.arguments = [mock_arg_0]
    
    result_0 = translator.xpath_eq_function(xpath_expr, function_0)
    xpath_expr.add_post_condition.assert_called_with('position() = 1')
    assert result_0 == xpath_expr

    # Test case 2: Valid numeric argument (index 5)
    mock_arg_5 = MagicMock()
    mock_arg_5.value = '5'
    function_5 = MagicMock()
    function_5.argument_types.return_value = ['NUMBER']
    function_5.arguments = [mock_arg_5]
    
    result_5 = translator.xpath_eq_function(xpath_expr, function_5)
    xpath_expr.add_post_condition.assert_called_with('position() = 6')
    assert result_5 == xpath_expr

    # Test case 3: Invalid argument type (STRING instead of NUMBER)
    mock_arg_str = MagicMock()
    mock_arg_str.value = 'abc'
    function_invalid = MagicMock()
    function_invalid.argument_types.return_value = ['STRING']
    function_invalid.arguments = [mock_arg_str]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, function_invalid)
    assert "Expected a single integer for :eq()" in str(excinfo.value)

    # Test case 4: Invalid argument type (IDENT instead of NUMBER)
    mock_arg_ident = MagicMock()
    mock_arg_ident.value = 'some_id'
    function_invalid_ident = MagicMock()
    function_invalid_ident.argument_types.return_value = ['IDENT']
    function_invalid_ident.arguments = [mock_arg_ident]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath_expr, function_invalid_ident)
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # 1. Test valid input: :lt(1) should result in position() < 2
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock(value='1')]
    
    result = translator.xpath_lt_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_once_with('position() < 2')

    # 2. Test invalid input type: :lt("string") should raise ExpressionError
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [MagicMock(value='abc')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function_invalid)
    
    assert "Expected a single integer for :gt(), got" in str(excinfo.value)
    # Note: The original code contains a typo in the error message using ':gt()' instead of ':lt()'
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock the function object passed to xpath_contains_function
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.argument_types.return_value = ['STRING']
    mock_func.arguments = [MagicMock(value='"test"')]
    
    # Mock cssselect's xpath_literal to return a simple string for the test
    translator.xpath_literal = MagicMock(return_value='"test"')
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert 'contains(., "test")' in str(xpath_expr)
    
    # Reset for next case
    xpath_expr = XPathExpr(path='div')

    # Case 2: Valid IDENT argument
    mock_func.argument_types.return_value = ['IDENT']
    mock_func.arguments = [MagicMock(value='some_id')]
    translator.xpath_literal = MagicMock(return_value='some_id')
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    assert 'contains(., some_id)' in str(xpath_expr)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func.argument_types.return_value = ['NUMBER']
    mock_func.arguments = [MagicMock(value='123')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath_expr, mock_func)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mocking the function object passed by cssselect
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock(value='"target_text"')]
    # We mock xpath_literal to return the same value for simplicity in testing logic flow
    translator.xpath_literal = MagicMock(side_effect=lambda x: x)
    
    result = translator.xpath_contains_function(xpath_expr, mock_function)
    
    assert result == xpath_expr
    assert 'contains(., "target_text")' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments = [MagicMock(value='some_id')]
    translator.xpath_literal = MagicMock(side_effect=lambda x: f"'{x}'")
    
    xpath_expr_2 = XPathExpr(path='div')
    translator.xpath_contains_function(xpath_expr_2, mock_function)
    assert "contains(., 'some_id')" in str(xpath_expr_2)

    # Case 3: Invalid argument type (e.g., NUMBER)
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock(value='123')]
    
    xpath_expr_3 = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_3, mock_function)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., BOOLEAN/other)
    mock_function.argument_types.return_value = ['BOOLEAN']
    xpath_expr_4 = XPathExpr(path='div')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath_expr_4, mock_function)
    assert "Expected a single string or ident" in str(excinfo.value)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mock the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Mock the function argument object with a value
    mock_function = MagicMock()
    mock_arg = MagicMock()
    mock_arg.value = '.bar'
    mock_function.arguments = [mock_arg]
    
    # Case 1: Valid STRING/IDENT argument
    mock_function.argument_types.return_value = ['STRING']
    
    # We need to mock css_to_xpath because it is called within xpath_has_function
    # Assuming the implementation of css_to_xpath works for our test input
    translator.css_to_xpath = MagicMock(return_value='descendant::.bar')
    
    result = translator.xpath_has_function(xpath_expr, mock_function)
    
    # Verify result is the xpath object itself
    assert result == xpath_expr
    # Verify add_post_condition was called with the translated XPath
    xpath_expr.add_post_condition.assert_called_with('descendant::.bar')

    # Case 2: Valid IDENT argument
    mock_function.argument_types.return_value = ['IDENT']
    xpath_expr.add_post_condition.reset_mock()
    
    result = translator.xpath_has_function(xpath_expr, mock_function)
    assert result == xpath_expr
    xpath_expr.add_post_condition.assert_called_with('descendant::.bar')

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, mock_function)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Case 4: Testing with a different selector value
    mock_arg.value = 'div'
    mock_function.argument_types.return_value = ['IDENT']
    translator.css_to_xpath.return_value = 'descendant::div'
    
    xpath_expr.add_post_condition.reset_mock()
    translator.xpath_has_function(xpath_expr, mock_function)
    xpath_expr.add_post_condition.assert_called_with('descendant::div')
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument
    mock_arg = MagicMock()
    mock_arg.value = '1'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    # Test valid :gt(1) -> position() > 2
    result = translator.xpath_gt_function(xpath, mock_function)
    assert result == xpath
    assert 'position() > 2' in str(xpath)
    
    # Reset for next test case
    xpath = XPathExpr(path='div')
    
    # Mock for an invalid argument type (e.g., STRING)
    mock_arg_invalid = MagicMock()
    mock_arg_invalid.value = 'foo'
    
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_invalid]
    
    # Test invalid argument type raises ExpressionError
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_function_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)

    # Test edge case: index 0 -> position() > 1
    xpath = XPathExpr(path='div')
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_function_zero.arguments = [mock_arg_zero]
    
    translator.xpath_gt_function(xpath, mock_function_zero)
    assert 'position() > 1' in str(xpath)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')

    # Mock function object for a valid NUMBER argument
    mock_func_valid = MagicMock()
    mock_func_valid.argument_types.return_value = ['NUMBER']
    mock_arg_valid = MagicMock()
    mock_arg_valid.value = '0'
    mock_func_valid.arguments = [mock_arg_valid]

    # Test valid case: :gt(0) should result in position() > 1
    result_valid = translator.xpath_gt_function(xpath, mock_func_valid)
    assert result_valid == xpath
    assert 'position() > 1' in str(xpath)

    # Test valid case: :gt(2) should result in position() > 3
    mock_arg_other = MagicMock()
    mock_arg_other.value = '2'
    mock_func_valid.arguments = [mock_arg_other]
    xpath_new = XPathExpr(path='p')
    translator.xpath_gt_function(xpath_new, mock_func_valid)
    assert 'position() > 3' in str(xpath_new)

    # Test error case: Invalid argument type (e.g., STRING instead of NUMBER)
    mock_func_invalid = MagicMock()
    mock_func_invalid.argument_types.return_value = ['STRING']
    mock_arg_invalid = MagicMock()
    mock_arg_invalid.value = 'abc'
    mock_func_invalid.arguments = [mock_arg_invalid]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, mock_func_invalid)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='p')
    
    # Mock for a valid NUMBER argument (e.g., :lt(1))
    mock_arg = MagicMock()
    mock_arg.value = '1'
    
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [mock_arg]

    # Test successful execution
    # :lt(1) should result in position() < (1 + 1) => position() < 2
    result = translator.xpath_lt_function(xpath_expr, mock_function)
    assert result == xpath_expr
    assert 'position() < 2' in str(xpath_expr)

    # Test error when argument type is not NUMBER
    mock_arg_string = MagicMock()
    mock_arg_string.value = 'text'
    
    mock_function_invalid = MagicMock()
    mock_function_invalid.argument_types.return_value = ['STRING']
    mock_function_invalid.arguments = [mock_arg_string]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath_expr, mock_function_invalid)
    assert "Expected a single integer for :gt(), got" in str(excinfo.value) or \
           "Expected a single integer for :lt(), got" in str(excinfo.value)

    # Test with index 0
    mock_arg_zero = MagicMock()
    mock_arg_zero.value = '0'
    mock_function_zero = MagicMock()
    mock_function_zero.argument_types.return_value = ['NUMBER']
    mock_function_zero.arguments = [mock_arg_zero]
    
    xpath_expr_zero = XPathExpr(path='div')
    translator.xpath_lt_function(xpath_expr_zero, mock_function_zero)
    assert 'position() < 1' in str(xpath_expr_zero)
```


# LLM-generated content at query #23
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
    
    # Mock css_to_xpath to return the expected XPath descendant axis
    translator.css_to_xpath = MagicMock(return_value='descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar")]')
    
    # Test valid string argument
    result = translator.xpath_has_function(xpath_expr, mock_function)
    assert result == xpath_expr
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar")]' in str(xpath_expr)

    # Test valid IDENT argument
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'div'
    translator.css_to_xpath.return_value = 'descendant::div'
    
    new_xpath_expr = XPathExpr(path='div', element='*')
    result = translator.xpath_has_function(new_xpath_expr, mock_function)
    assert 'descendant::div' in str(new_xpath_expr)

    # Test invalid argument type (NUMBER)
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(XPathExpr(), mock_function)

    # Test invalid argument type (LIST)
    mock_function.argument_types = ['STRING', 'STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    xpath_expr = MagicMock(spec=XPathExpr)
    
    # Helper to create mock function arguments
    def create_mock_func(arg_type, value):
        func = MagicMock()
        func.argument_types = [arg_type]
        arg = MagicMock()
        arg.value = value
        func.arguments = [arg]
        return func

    # Mock css_to_xpath to return a predictable string
    translator.css_to_xpath = MagicMock(side_effect=lambda selector, prefix='': f"{prefix}{selector}")

    # Test Case 1: Valid STRING argument (e.g., :has(".bar"))
    func_string = create_mock_func('STRING', '.bar')
    translator.xpath_has_function(xpath_expr, func_string)
    xpath_expr.add_post_condition.assert_called_with('descendant::.bar')

    # Test Case 2: Valid IDENT argument (e.g., :has(div))
    func_ident = create_mock_func('IDENT', 'div')
    translator.xpath_has_function(xpath_expr, func_ident)
    xpath_expr.add_post_condition.assert_called_with('descendant::div')

    # Test Case 3: Invalid argument type (e.g., NUMBER)
    func_invalid = create_mock_func('NUMBER', '123')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, func_invalid)
    assert "Expected a single string or ident" in str(excinfo.value)

    # Test Case 4: Invalid argument type (e.g., BOOLEAN)
    func_bool = create_mock_func('BOOLEAN', 'true')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(xpath_expr, func_bool)
    assert "Expected a single string or ident" in str(excinfo.value)
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mocking the function object from cssselect
    mock_function = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_function.argument_types = ['STRING']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = "'hello'"
    
    result = translator.xpath_contains_function(xpath, mock_function)
    assert result == xpath
    # Check if the post condition was added correctly
    # Note: cssselect's literal wrapping might vary; assuming standard string formatting
    assert 'contains(., \'hello\')' in str(xpath)

    # Case 2: Valid IDENT argument
    mock_function.argument_types = ['IDENT']
    mock_function.arguments[0].value = 'some_id'
    
    # Resetting xpath for a clean test of the second condition
    new_xpath = XPathExpr(path='div')
    translator.xpath_contains_function(new_xpath, mock_function)
    # Using contains logic check (checking if string part exists)
    assert 'contains(., some_id)' in str(new_xpath) or "contains(., 'some_id')" in str(new_xpath)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_function.argument_types = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)

    # Case 4: Invalid argument type (e.g., LIST) should raise ExpressionError
    mock_function.argument_types = ['LIST']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(XPathExpr(), mock_function)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr(path='div')
    
    # Mock for a valid NUMBER argument in cssselect
    mock_arg = MagicMock()
    mock_arg.value = '0'
    
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [mock_arg]

    # Test successful execution: :gt(0) should result in position() > 1
    result = translator.xpath_gt_function(xpath, mock_function)
    assert result == xpath
    assert 'position() > 1' in str(xpath)

    # Test with a different number: :gt(2) should result in position() > 3
    mock_arg.value = '2'
    xpath_new = XPathExpr(path='p')
    translator.xpath_gt_function(xpath_new, mock_function)
    assert 'position() > 3' in str(xpath_new)

    # Test error handling for non-NUMBER argument type
    mock_func_error = MagicMock()
    mock_func_error.argument_types = ['STRING']
    mock_func_error.arguments = [mock_arg]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(XPathExpr(), mock_func_error)
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Mock the XPathExpr object that would be passed to the function
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Helper to create a mock function argument
    def create_mock_arg(value, arg_types):
        arg = MagicMock()
        arg.value = value
        return arg, arg_types

    # Case 1: Valid STRING argument
    arg_val, arg_types = create_mock_arg('"div"', ['STRING'])
    mock_func = MagicMock()
    mock_func.arguments = [arg_val]
    mock_func.argument_types.return_value = arg_types
    
    # Mock css_to_xpath to return a specific string
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(mock_xpath, mock_func)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::div')

    # Case 2: Valid IDENT argument
    arg_val, arg_types = create_mock_arg('div', ['IDENT'])
    mock_func.arguments = [arg_val]
    mock_func.argument_types.return_value = arg_types
    
    translator.css_to_xpath = MagicMock(return_value='descendant::div')
    
    result = translator.xpath_has_function(mock_xpath, mock_func)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('descendant::div')

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    arg_val, arg_types = create_mock_arg(123, ['NUMBER'])
    mock_func.arguments = [arg_val]
    mock_func.argument_types.return_value = arg_types
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_has_function(mock_xpath, mock_func)
    
    assert "Expected a single string or ident for :has()" in str(excinfo.value)
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from cssselect.xpath import ExpressionError

def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mocking the XPathExpr object
    mock_xpath = MagicMock(spec=XPathExpr)
    
    # Case 1: Valid NUMBER argument (index 0 -> position() > 1)
    mock_arg = MagicMock()
    mock_arg.value = '0'
    
    mock_function = MagicMock()
    mock_function.argument_types = ['NUMBER']
    mock_function.arguments = [mock_arg]
    
    result = translator.xpath_gt_function(mock_xpath, mock_function)
    
    assert result == mock_xpath
    mock_xpath.add_post_condition.assert_called_with('position() > 1')

    # Case 2: Valid NUMBER argument (index 5 -> position() > 6)
    mock_arg.value = '5'
    translator.xpath_gt_function(mock_xpath, mock_function)
    mock_xpath.add_post_condition.assert_called_with('position() > 6')

    # Case 3: Invalid argument type (STRING instead of NUMBER)
    mock_function.argument_types = ['STRING']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(mock_xpath, mock_function)
    
    assert "Expected a single integer for :gt()" in str(excinfo.value)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath_expr = XPathExpr(path='div')
    
    # Mock the function object passed by cssselect
    mock_func = MagicMock()
    
    # Case 1: Valid STRING argument
    mock_func.argument_types.return_value = ['STRING']
    mock_func.arguments = [MagicMock(value='"hello"')]
    # Mocking xpath_literal behavior (assuming it returns the string as is or quoted)
    translator.xpath_literal = MagicMock(return_value='"hello"')
    
    result = translator.xpath_contains_function(xpath_expr, mock_func)
    
    assert result == xpath_expr
    assert 'contains(., "hello")' in str(xpath_expr)

    # Case 2: Valid IDENT argument
    mock_func.argument_types.return_value = ['IDENT']
    mock_func.arguments = [MagicMock(value='some_id')]
    translator.xpath_literal = MagicMock(return_value='some_id')
    
    # Reset xpath_expr for a clean test on post-condition
    xpath_expr_2 = XPathExpr(path='div')
    translator.xpath_contains_function(xpath_expr_2, mock_func)
    assert 'contains(., some_id)' in str(xpath_expr_2)

    # Case 3: Invalid argument type (e.g., NUMBER) should raise ExpressionError
    mock_func.argument_types.return_value = ['NUMBER']
    mock_func.arguments = [MagicMock(value='123')]
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)

    # Case 4: Invalid argument type (e.g., BOOLEAN) should raise ExpressionError
    mock_func.argument_types.return_value = ['BOOLEAN']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(XPathExpr(), mock_func)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)
```


# LLM-generated content at query #30
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

    # Test valid :eq(0) -> position() = 1
    result_xpath = translator.xpath_eq_function(xpath, mock_function)
    assert result_xpath == xpath
    assert 'position() = 1' in str(xpath)

    # Test valid :eq(5) -> position() = 6
    mock_arg_value.value = '5'
    xpath_two = XPathExpr(path='p')
    translator.xpath_eq_function(xpath_two, mock_function)
    assert 'position() = 6' in str(xpath_two)

    # Test invalid argument type (e.g., STRING)
    mock_function.argument_types.return_value = ['STRING']
    mock_arg_string = MagicMock()
    mock_arg_string.value = '"not-a-number"'
    mock_function.arguments = [mock_arg_string]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, mock_function)
    assert "Expected a single integer for :eq()" in str(excinfo.value)
```


