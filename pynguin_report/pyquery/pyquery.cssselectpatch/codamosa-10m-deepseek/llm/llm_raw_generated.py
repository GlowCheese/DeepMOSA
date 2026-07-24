####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='/div/p', element='p', condition='', star_prefix=False)
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '/div/p[position() > 1]'


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with a simple selector that should match
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test with an ident argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with no match
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.nonexistent'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " nonexistent ")]' in str(xpath)
    
    # Test that it raises ExpressionError for invalid arguments
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '5'})()]
            })()
        )
```


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='title')]
    translator.xpath_literal = MagicMock(return_value="'title'")
    
    result = translator.xpath_contains_function(xpath, function)
    
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    function2 = MagicMock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments = [MagicMock(value='text')]
    translator.xpath_literal = MagicMock(return_value="'text'")
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    
    assert result2.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    function3 = MagicMock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [MagicMock(value=5)]
    
    with pytest.raises(ExpressionError) as exc_info:
        translator.xpath_contains_function(xpath3, function3)
    assert "Expected a single string or ident" in str(exc_info.value)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a valid number argument
    xpath = translator.xpathexpr_cls()
    selector = "h1"
    
    # Mock the function object
    class MockArg:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['NUMBER']
    
    function = MockFunction([MockArg("0")])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == "position() = 1"
    
    # Test with index 1 (second element)
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction([MockArg("1")])
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == "position() = 2"
    
    # Test with larger index
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction([MockArg("5")])
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == "position() = 6"
    
    # Test with negative index (should still work since we don't validate)
    xpath4 = translator.xpathexpr_cls()
    function4 = MockFunction([MockArg("-1")])
    result4 = translator.xpath_eq_function(xpath4, function4)
    assert result4.post_condition == "position() = 0"
    
    # Test that it raises ExpressionError for non-number argument
    class MockStringFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['STRING']
    
    import pytest
    with pytest.raises(ExpressionError):
        xpath5 = translator.xpathexpr_cls()
        function5 = MockStringFunction([MockArg("test")])
        translator.xpath_eq_function(xpath5, function5)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Mock function object for testing
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '0'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with different index
    xpath2 = translator.xpathexpr_cls()
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('obj', (object,), {'value': '2'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() = 3'


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has function with class selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(), 
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test has function with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(), 
        type('Function', (), {
            'argument_types': lambda: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has function raises error with invalid arguments
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(), 
            type('Function', (), {
                'argument_types': lambda: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '5'})()]
            })()
        )
```


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with positive integer
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative integer
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)
    
    # Test that it raises ExpressionError for empty arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: [],
        'arguments': []
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr('div', 'div', ''),
        type('obj', (object,), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('arg', (object,), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test that it raises ExpressionError for invalid arguments
    try:
        translator.xpath_has_function(
            XPathExpr('div', 'div', ''),
            type('obj', (object,), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('arg', (object,), {'value': '123'})()]
            })()
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with IDENT type
    xpath2 = translator.xpath_has_function(
        XPathExpr('div', 'div', ''),
        type('obj', (object,), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('arg', (object,), {'value': 'bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath2)
    assert 'bar' in str(xpath2)


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1', element='h1')
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, Number
    function = Function('eq', [Number('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//h1[position() = 1]'
    
    # Test with different index values
    xpath2 = translator.xpathexpr_cls(path='//h1', element='h1')
    function2 = Function('eq', [Number('2')])
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert str(result2) == '//h1[position() = 3]'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    from cssselect.parser import String
    function3 = Function('eq', [String('test')])
    xpath3 = translator.xpathexpr_cls(path='//h1', element='h1')
    try:
        translator.xpath_eq_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it preserves post_condition from previous calls
    xpath4 = translator.xpathexpr_cls(path='//h1', element='h1')
    xpath4.add_post_condition('@class')
    function4 = Function('eq', [Number('1')])
    result4 = translator.xpath_eq_function(xpath4, function4)
    assert 'position() = 2' in str(result4)
    assert '@class' in str(result4)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple case where eq(0) should match first element
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with eq(1) should match second element
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with negative index
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    with pytest.raises(ExpressionError):
        function_invalid = type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
        translator.xpath_eq_function(translator.xpathexpr_cls(), function_invalid)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Create a mock function with NUMBER argument type
    class MockFunction:
        def __init__(self, value):
            self.arguments = [type('MockArgument', (), {'value': str(value)})()]
            
        def argument_types(self):
            return ['NUMBER']
    
    # Test with index 0 (first element)
    xpath = XPathExpr('div', 'p', '')
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1 (second element)
    xpath = XPathExpr('div', 'p', '')
    result = translator.xpath_eq_function(xpath, MockFunction(1))
    assert result.post_condition == 'position() = 2'
    
    # Test with index 5 (sixth element)
    xpath = XPathExpr('div', 'p', '')
    result = translator.xpath_eq_function(xpath, MockFunction(5))
    assert result.post_condition == 'position() = 6'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockFunctionInvalid:
        def __init__(self):
            self.arguments = ['string']
            
        def argument_types(self):
            return ['STRING']
    
    import pytest
    with pytest.raises(ExpressionError):
        xpath = XPathExpr('div', 'p', '')
        translator.xpath_eq_function(xpath, MockFunctionInvalid())
```


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a valid NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArg', (), {'value': '0'})()]
    
    mock_xpath = XPathExpr()
    result = translator.xpath_eq_function(mock_xpath, MockFunction())
    assert result.post_condition == 'position() = 1', "Expected position() = 1 for eq(0)"
    
    # Test with another index value
    mock_xpath2 = XPathExpr()
    MockFunction.arguments = [type('MockArg', (), {'value': '2'})()]
    result2 = translator.xpath_eq_function(mock_xpath2, MockFunction())
    assert result2.post_condition == 'position() = 3', "Expected position() = 3 for eq(2)"
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArg', (), {'value': 'test'})()]
    
    mock_xpath3 = XPathExpr()
    try:
        translator.xpath_eq_function(mock_xpath3, MockFunctionInvalid())
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with valid string argument
    xpath = translator.xpathexpr_cls()
    from cssselect.parser import Function, parse
    # Mock a function call with string argument
    function = Function('has', ['".bar"'])
    function.argument_types = lambda: ['STRING']
    function.arguments = [type('arg', (), {'value': '.bar'})()]
    
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::' in str(result)
    assert 'bar' in str(result)
    
    # Test with valid ident argument
    xpath2 = translator.xpathexpr_cls()
    function2 = Function('has', ['div'])
    function2.argument_types = lambda: ['IDENT']
    function2.arguments = [type('arg', (), {'value': 'div'})()]
    
    result2 = translator.xpath_has_function(xpath2, function2)
    assert 'descendant::' in str(result2)
    assert 'div' in str(result2)
    
    # Test with invalid argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = Function('has', ['123'])
    function3.argument_types = lambda: ['NUMBER']
    function3.arguments = [type('arg', (), {'value': '123'})()]
    
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with unsupported argument type list
    xpath4 = translator.xpathexpr_cls()
    function4 = Function('has', ['a', 'b'])
    function4.argument_types = lambda: ['STRING', 'STRING']
    function4.arguments = [type('arg', (), {'value': 'a'})(), type('arg', (), {'value': 'b'})()]
    
    try:
        translator.xpath_has_function(xpath4, function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is added correctly
    xpath5 = translator.xpathexpr_cls(path='//div', element='div')
    function5 = Function('has', ['span'])
    function5.argument_types = lambda: ['IDENT']
    function5.arguments = [type('arg', (), {'value': 'span'})()]
    
    result5 = translator.xpath_has_function(xpath5, function5)
    result_str = str(result5)
    assert 'descendant::' in result_str
    assert 'span' in result_str
    assert '//div' in result_str
```


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with simple selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert "descendant::" in str(xpath)
    
    # Test with tag selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert "descendant::" in str(xpath)
    
    # Test with IDENT type argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert "descendant::" in str(xpath)
    
    # Test error handling for invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '42'})()]
            })()
        )
```


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    xpath.element = 'h1'
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'
    
    # Test with negative number
    xpath2 = translator.xpathexpr_cls()
    xpath2.element = 'h1'
    
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '-1'})()]
    
    result2 = translator.xpath_lt_function(xpath2, MockFunctionNegative())
    assert result2.post_condition == 'position() < 0'
    
    # Test with zero
    xpath3 = translator.xpathexpr_cls()
    xpath3.element = 'h1'
    
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result3 = translator.xpath_lt_function(xpath3, MockFunctionZero())
    assert result3.post_condition == 'position() < 1'
    
    # Test that non-number argument raises ExpressionError
    xpath4 = translator.xpathexpr_cls()
    
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = ['foo']
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, MockFunctionInvalid())


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    function2 = type('MockFunction', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('MockArgument', (), {'value': 'content'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test with invalid argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '5'})()]
    })()
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, function3)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr()
    xpath.element = 'h1'
    xpath.path = '//h1'
    
    # Mock a function with NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    
    # Verify the post_condition is correct
    assert result.post_condition == 'position() > 1'


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='@class="foo"'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()],
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='@class="foo"'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()],
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='@class="foo"'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'bar'})()],
        })()
    )
    assert 'descendant::bar' in str(xpath)
    
    # Test invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()],
            })()
        )


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    from unittest.mock import Mock, MagicMock
    translator = JQueryTranslator()
    
    # Create a mock XPathExpr
    xpath = XPathExpr()
    
    # Create a mock function with NUMBER argument type
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '0'
    
    # Test with value 0 (should result in position() = 1)
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with value 2 (should result in position() = 3)
    xpath2 = XPathExpr()
    function.arguments[0].value = '2'
    result2 = translator.xpath_eq_function(xpath2, function)
    assert result2.post_condition == 'position() = 3'
    
    # Test with value 100 (should result in position() = 101)
    xpath3 = XPathExpr()
    function.arguments[0].value = '100'
    result3 = translator.xpath_eq_function(xpath3, function)
    assert result3.post_condition == 'position() = 101'
    
    # Test with negative value (-1 should result in position() = 0)
    xpath4 = XPathExpr()
    function.arguments[0].value = '-1'
    result4 = translator.xpath_eq_function(xpath4, function)
    assert result4.post_condition == 'position() = 0'


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with proper numeric argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another numeric value
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '3'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 4'
    
    # Test that non-numeric argument raises ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    function = parse(':contains("test")')[0].pseudo_class
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'test')" in str(result)
    
    # Test with ident argument
    xpath2 = translator.xpathexpr_cls()
    function2 = parse(':contains(test)')[0].pseudo_class
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert "contains(., 'test')" in str(result2)
    
    # Test that invalid argument type raises ExpressionError
    xpath3 = translator.xpathexpr_cls()
    from cssselect.parser import Function as Func
    invalid_func = Func('contains', [])
    try:
        translator.xpath_contains_function(xpath3, invalid_func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a mock XPathExpr object
    class MockXPath:
        def __init__(self):
            self.post_condition = None
            
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    # Test with value 0
    xpath = MockXPath()
    result = translator.xpath_lt_function(xpath, MockFunction('0'))
    assert result.post_condition == 'position() < 1'
    
    # Test with value 1
    xpath = MockXPath()
    result = translator.xpath_lt_function(xpath, MockFunction('1'))
    assert result.post_condition == 'position() < 2'
    
    # Test with value 5
    xpath = MockXPath()
    result = translator.xpath_lt_function(xpath, MockFunction('5'))
    assert result.post_condition == 'position() < 6'
    
    # Test that it raises ExpressionError for non-numeric arguments
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(MockXPath(), MockFunction('abc'))


class MockFunction:
    def __init__(self, value):
        self.arguments = [MockArgument(value)]
    
    def argument_types(self):
        return ['NUMBER']


class MockArgument:
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    assert result is xpath
    
    # Test with zero index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test with non-number argument type
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_lt_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'}), type('Arg', (), {'value': '2'})()]
    })()
    try:
        translator.xpath_lt_function(xpath5, function5)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is properly set when there's an existing condition
    xpath6 = translator.xpathexpr_cls()
    xpath6.add_post_condition('existing_condition')
    function6 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result6 = translator.xpath_lt_function(xpath6, function6)
    assert 'existing_condition and (position() < 3)' in result6.post_condition


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another number
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    with pytest.raises(ExpressionError):
        function_invalid = type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
        translator.xpath_gt_function(XPathExpr(), function_invalid)


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test text'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'test text')"
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'identifier'})()]
    
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'identifier')"
    
    # Test that it raises ExpressionError for invalid argument types
    xpath3 = translator.xpathexpr_cls()
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '42'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, MockFunctionInvalid())
    
    # Test that it raises ExpressionError for multiple arguments
    xpath4 = translator.xpathexpr_cls()
    class MockFunctionMultiple:
        def argument_types(self):
            return ['STRING', 'STRING']
        arguments = [
            type('MockArgument', (), {'value': 'first'}),
            type('MockArgument', (), {'value': 'second'})
        ]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath4, MockFunctionMultiple())
```


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument type
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
    
    # Test with post_condition already set
    xpath = XPathExpr(path='//div', element='div')
    xpath.post_condition = 'position() = 1'
    result = translator.xpath_has_function(
        xpath,
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'position() = 1' in str(result)
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(result)
```


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath object
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='0')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with different index
    xpath2 = XPathExpr()
    function2 = Mock()
    function2.argument_types.return_value = ['NUMBER']
    function2.arguments = [Mock(value='2')]
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative index
    xpath3 = XPathExpr()
    function3 = Mock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [Mock(value='-1')]
    
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-number arguments
    xpath4 = XPathExpr()
    function4 = Mock()
    function4.argument_types.return_value = ['STRING']
    function4.arguments = [Mock(value='test')]
    
    try:
        translator.xpath_gt_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it returns the same xpath object
    xpath5 = XPathExpr()
    function5 = Mock()
    function5.argument_types.return_value = ['NUMBER']
    function5.arguments = [Mock(value='5')]
    
    result5 = translator.xpath_gt_function(xpath5, function5)
    assert result5 is xpath5


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Mock function object with NUMBER argument
    class MockFunction:
        class MockArgument:
            def __init__(self, value):
                self.value = value
        arguments = [MockArgument("2")]
        def argument_types(self):
            return ['NUMBER']
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]" in str(xpath)
    
    # Test with element selector
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert "descendant::div" in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert "descendant::span" in str(xpath)
    
    # Test that invalid argument type raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )
    
    # Test empty result scenario (no matching descendant)
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.nonexistent'})()]
        })()
    )
    assert "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' nonexistent ')]" in str(xpath)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_even_pseudo():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr(path='//p', element='p')
    result = translator.xpath_even_pseudo(xpath)
    
    assert result.post_condition == 'position() mod 2 = 1'
    assert result.path == '//p'
    assert result.element == 'p'
    
    # Test that the xpath string representation includes the post_condition
    str_result = str(result)
    assert '[position() mod 2 = 1]' in str_result
    
    # Test with an xpath that already has a condition
    xpath_with_condition = XPathExpr(path='//div', element='div', condition='@class')
    result_with_condition = translator.xpath_even_pseudo(xpath_with_condition)
    assert result_with_condition.post_condition == 'position() mod 2 = 1'
    assert result_with_condition.condition == '@class'
    
    # Test the actual behavior with a mock selector
    xpath2 = XPathExpr(path='//*', element='*')
    result2 = translator.xpath_even_pseudo(xpath2)
    assert result2.post_condition == 'position() mod 2 = 1'
    
    # Verify the complete xpath string
    full_xpath = str(result2)
    assert full_xpath == '//*[position() mod 2 = 1]'


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with another index value
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '3'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 4'
    
    # Test with negative index
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    import pytest
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['NUMBER'], arguments=[Mock(value='2')]))
    assert str(result) == '*[position() < 3]'
    
    # Test with edge case: value = 0
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['NUMBER'], arguments=[Mock(value='0')]))
    assert str(result) == '*[position() < 1]'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['NUMBER'], arguments=[Mock(value='-1')]))
    assert str(result) == '*[position() < 0]'
    
    # Test with element specified
    xpath = translator.xpathexpr_cls(element='div')
    result = translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['NUMBER'], arguments=[Mock(value='1')]))
    assert str(result) == 'div[position() < 2]'
    
    # Test with condition already present
    xpath = translator.xpathexpr_cls(condition='@class = "test"')
    result = translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['NUMBER'], arguments=[Mock(value='3')]))
    assert str(result) == '*[@class = "test"][position() < 4]'
    
    # Test that it raises ExpressionError for non-number arguments
    from cssselect.xpath import ExpressionError
    try:
        xpath = translator.xpathexpr_cls()
        translator.xpath_lt_function(xpath, Mock(function=lambda: None, argument_types=lambda: ['STRING'], arguments=[Mock(value='test')]))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with string argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'title'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert "contains(., 'title')" in str(result)
    
    # Test with ident argument
    translator2 = JQueryTranslator()
    xpath2 = translator2.xpathexpr_cls()
    
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'content'})()]
    
    result2 = translator2.xpath_contains_function(xpath2, MockFunctionIdent())
    assert "contains(., 'content')" in str(result2)
    
    # Test with invalid argument type
    translator3 = JQueryTranslator()
    xpath3 = translator3.xpathexpr_cls()
    
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '42'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator3.xpath_contains_function(xpath3, MockFunctionInvalid())


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockNumberFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '0'})()]
    
    result = translator.xpath_eq_function(xpath, MockNumberFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    class MockNumberFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '5'})()]
    
    result2 = translator.xpath_eq_function(xpath2, MockNumberFunction2())
    assert result2.post_condition == 'position() = 6'
    
    # Test that it raises ExpressionError for invalid argument types
    class MockStringFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(translator.xpathexpr_cls(), MockStringFunction())
```


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = XPathExpr()
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'
    
    # Test with zero
    xpath = XPathExpr()
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    result = translator.xpath_lt_function(xpath, MockFunctionZero())
    assert result.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath = XPathExpr()
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '-1'})()]
    result = translator.xpath_lt_function(xpath, MockFunctionNegative())
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type (STRING instead of NUMBER)
    xpath = XPathExpr()
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    try:
        translator.xpath_lt_function(xpath, MockFunctionString())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath = XPathExpr()
    class MockFunctionMultiple:
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    try:
        translator.xpath_lt_function(xpath, MockFunctionMultiple())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that original xpath is returned with post_condition set
    xpath = XPathExpr(condition="test_condition")
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '5'})()]
    result = translator.xpath_lt_function(xpath, MockFunction2())
    assert result is xpath
    assert result.post_condition == 'position() < 6'
    assert result.condition == "test_condition"  # Original condition preserved


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    assert 'contains' not in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test that invalid argument type raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with a simple selector
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(result)
    
    # Test with an element selector
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::div' in str(result)
    
    # Test that post_condition is added
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    
    # Test with IDENT argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::div' in str(result)
    
    # Test that invalid argument types raise ExpressionError
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple XPath for div elements
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//div[position() = 1]'
    assert result.post_condition == 'position() = 1'

    # Test with index 2
    xpath2 = translator.xpathexpr_cls(path='//p', element='p')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert str(result2) == '//p[position() = 3]'
    assert result2.post_condition == 'position() = 3'

    # Test with negative number (should still work as position comparison)
    xpath3 = translator.xpathexpr_cls(path='//span', element='span')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert str(result3) == '//span[position() = 0]'
    assert result3.post_condition == 'position() = 0'

    # Test that it raises ExpressionError for invalid argument types
    invalid_function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    
    try:
        translator.xpath_eq_function(xpath, invalid_function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    class MockXPath:
        def __init__(self):
            self.post_condition = None
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    xpath = MockXPath()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3', "Expected position() = 3 for eq(2)"
    
    # Test with first element (index 0)
    class MockFunctionFirst:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    xpath2 = MockXPath()
    result2 = translator.xpath_eq_function(xpath2, MockFunctionFirst())
    assert result2.post_condition == 'position() = 1', "Expected position() = 1 for eq(0)"
    
    # Test with non-NUMBER argument type should raise ExpressionError
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    try:
        translator.xpath_eq_function(MockXPath(), MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple XPath
    xpath = translator.xpathexpr_cls()
    xpath = translator.xpath_lt_function(xpath, MockFunction('1'))
    assert str(xpath) == '*[position() < 2]'
    
    # Test with a complex XPath
    xpath = translator.xpathexpr_cls(path='//div', element='div', condition='@class')
    xpath = translator.xpath_lt_function(xpath, MockFunction('3'))
    assert str(xpath) == '//div[@class][position() < 4]'
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    xpath = translator.xpath_lt_function(xpath, MockFunction('0'))
    assert str(xpath) == '*[position() < 1]'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    xpath = translator.xpath_lt_function(xpath, MockFunction('-1'))
    assert str(xpath) == '*[position() < 0]'


class MockFunction:
    def __init__(self, value):
        self.arguments = [MockArgument(value)]
    
    def argument_types(self):
        return ['NUMBER']


class MockArgument:
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), 
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'title'})()]
        })()
    )
    assert 'contains(., "title")' in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'text'})()]
        })()
    )
    assert 'contains(., "text")' in str(xpath)
    
    # Test raises ExpressionError for invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
    
    # Test with multiple arguments (should raise error)
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['STRING', 'STRING'],
                'arguments': [type('Argument', (), {'value': 'a'}),
                            type('Argument', (), {'value': 'b'})]
            })()
        )
```


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"
    
    # Test with ident argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition == "descendant::div"
    
    # Test with invalid argument type
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with empty selector
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': ''})()]
    })()
    result4 = translator.xpath_has_function(xpath4, function4)
    assert result4.post_condition == "descendant::*"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_lt_function(xpath, function)
    
    assert result.post_condition == 'position() < 3'
    
    # Test with different value
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_lt_function(xpath, function)
    
    assert result.post_condition == 'position() < 1'
    
    # Test that it raises ExpressionError for non-number arguments
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and positive number
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction(['NUMBER'], ['1']))
    assert result.post_condition == 'position() > 2', "Expected position() > 2 for :gt(1)"
    
    # Test with zero index
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction(['NUMBER'], ['0']))
    assert result.post_condition == 'position() > 1', "Expected position() > 1 for :gt(0)"
    
    # Test with negative number
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction(['NUMBER'], ['-1']))
    assert result.post_condition == 'position() > 0', "Expected position() > 0 for :gt(-1)"
    
    # Test with large number
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction(['NUMBER'], ['100']))
    assert result.post_condition == 'position() > 101', "Expected position() > 101 for :gt(100)"
    
    # Test with invalid argument type (should raise ExpressionError)
    with pytest.raises(ExpressionError) as excinfo:
        xpath = XPathExpr()
        translator.xpath_gt_function(xpath, MockFunction(['STRING'], ['test']))
    assert "Expected a single integer for :gt()" in str(excinfo.value)
    
    # Test with multiple arguments (should still raise error since we check types)
    with pytest.raises(ExpressionError) as excinfo:
        xpath = XPathExpr()
        translator.xpath_gt_function(xpath, MockFunction(['NUMBER', 'NUMBER'], ['1', '2']))
    assert "Expected a single integer for :gt()" in str(excinfo.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Mock an XPathExpr to pass to the method
    xpath = XPathExpr()
    
    # Create a mock function object
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        
        class arguments:
            def __init__(self):
                self.value = '0'
    
    mock_func = MockFunction()
    mock_func.arguments = [type('MockArgument', (), {'value': '0'})()]
    
    # Test with index 0 (first element)
    result = translator.xpath_eq_function(xpath, mock_func)
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1 (second element)
    xpath2 = XPathExpr()
    mock_func.arguments[0].value = '1'
    result2 = translator.xpath_eq_function(xpath2, mock_func)
    assert result2.post_condition == 'position() = 2'
    
    # Test with negative index (should handle negative numbers)
    xpath3 = XPathExpr()
    mock_func.arguments[0].value = '-1'
    result3 = translator.xpath_eq_function(xpath3, mock_func)
    assert result3.post_condition == 'position() = 0'
    
    # Test that ExpressionError is raised for non-number arguments
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        
        class arguments:
            def __init__(self):
                self.value = 'test'
    
    mock_func_invalid = MockFunctionInvalid()
    mock_func_invalid.arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), mock_func_invalid)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with ident argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'content'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test that it raises ExpressionError for invalid argument types
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '5'})()]
    })()
    try:
        translator.xpath_contains_function(XPathExpr(), function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Arg', (), {'value': 'a'}), type('Arg', (), {'value': 'b'})]
    })()
    try:
        translator.xpath_contains_function(XPathExpr(), function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='title')]
    
    result = translator.xpath_contains_function(xpath, function)
    
    assert result.post_condition == "contains(., 'title')"
    assert result is xpath

    # Test with IDENT type
    xpath2 = XPathExpr()
    function2 = MagicMock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments = [MagicMock(value='test')]
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    
    assert result2.post_condition == "contains(., 'test')"

    # Test with invalid argument types
    xpath3 = XPathExpr()
    function3 = MagicMock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [MagicMock(value='42')]
    
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains"):
        translator.xpath_contains_function(xpath3, function3)


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '2'})()]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'
    
    # Test with zero index
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '0'})()]
    
    result2 = translator.xpath_lt_function(xpath2, MockFunctionZero())
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative index
    xpath3 = translator.xpathexpr_cls()
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '-1'})()]
    
    result3 = translator.xpath_lt_function(xpath3, MockFunctionNegative())
    assert result3.post_condition == 'position() < 0'


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpath_eq_function(XPathExpr(), MockFunction(['NUMBER'], '2'))
    assert str(xpath) == '*[position() = 3]'
    
    # Test with first element (index 0)
    xpath = translator.xpath_eq_function(XPathExpr(), MockFunction(['NUMBER'], '0'))
    assert str(xpath) == '*[position() = 1]'
    
    # Test with invalid argument type
    try:
        translator.xpath_eq_function(XPathExpr(), MockFunction(['STRING'], 'test'))
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    try:
        translator.xpath_eq_function(XPathExpr(), MockFunction(['NUMBER', 'NUMBER'], '1,2'))
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

class MockFunction:
    def __init__(self, argument_types, arguments):
        self._argument_types = argument_types
        self.arguments = MockArguments(arguments)
    
    def argument_types(self):
        return self._argument_types

class MockArguments:
    def __init__(self, value):
        self.value = MockValue(value)
    
    def __repr__(self):
        return self.value.value

class MockValue:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return self.value
```


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    from cssselect.parser import Function, Token
    func = Function('gt', [Token('NUMBER', '2')])
    result = translator.xpath_gt_function(xpath, func)
    assert result.post_condition == 'position() > 3'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('gt', [Token('NUMBER', '0')])
    result2 = translator.xpath_gt_function(xpath2, func2)
    assert result2.post_condition == 'position() > 1'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('gt', [Token('STRING', 'test')])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, func3)
    
    # Test that it raises ExpressionError for multiple arguments
    xpath4 = translator.xpathexpr_cls()
    func4 = Function('gt', [Token('NUMBER', '1'), Token('NUMBER', '2')])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath4, func4)


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = Mock(function_type='function', name='contains', 
                   arguments=[Mock(value='title', type='STRING')])
    function.argument_types = lambda: ['STRING']
    
    result = translator.xpath_contains_function(xpath, function)
    
    assert result.post_condition == "contains(., 'title')"
    
    # Test with ident argument  
    xpath = translator.xpathexpr_cls()
    function = Mock(function_type='function', name='contains',
                   arguments=[Mock(value='text', type='IDENT')])
    function.argument_types = lambda: ['IDENT']
    
    result = translator.xpath_contains_function(xpath, function)
    
    assert result.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = Mock(function_type='function', name='contains',
                   arguments=[Mock(value='123', type='NUMBER')])
    function.argument_types = lambda: ['NUMBER']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, parse
    from unittest.mock import MagicMock
    
    # Create a mock function with NUMBER argument type
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock()]
    mock_function.arguments[0].value = '0'
    
    # Create a mock xpath
    mock_xpath = MagicMock()
    mock_xpath.post_condition = None
    
    result = translator.xpath_eq_function(mock_xpath, mock_function)
    
    # Verify add_post_condition was called with correct position
    mock_xpath.add_post_condition.assert_called_once_with('position() = 1')
    assert result == mock_xpath
    
    # Test with index 1 (position 2)
    mock_function.arguments[0].value = '1'
    mock_xpath2 = MagicMock()
    mock_xpath2.post_condition = None
    
    result2 = translator.xpath_eq_function(mock_xpath2, mock_function)
    mock_xpath2.add_post_condition.assert_called_once_with('position() = 2')
    assert result2 == mock_xpath2
    
    # Test with invalid argument type (non-NUMBER)
    mock_function.argument_types.return_value = ['STRING']
    
    try:
        translator.xpath_eq_function(MagicMock(), mock_function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " test ")]'

    # Test with IDENT argument type
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition == 'descendant::div'

    # Test with invalid argument types
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

    # Test with no arguments
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: [],
        'arguments': []
    })()
    
    try:
        translator.xpath_has_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with different number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})]
    })()
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'title'})()]
        })()
    )
    assert 'contains(., "title")' in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//p', element='p'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'text'})()]
        })()
    )
    assert 'contains(., "text")' in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '42'})()]
            })()
        )
    
    # Test with multiple arguments
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['STRING', 'STRING'],
                'arguments': [
                    type('Argument', (), {'value': 'first'}),
                    type('Argument', (), {'value': 'second'})
                ]
            })()
        )


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function object
    class MockFunction:
        def __init__(self, arg_type, arg_value):
            self.arguments = [MockArgument(arg_value)]
            self.arg_type = arg_type
            
        def argument_types(self):
            return [self.arg_type]
    
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    # Test with STRING argument
    string_function = MockFunction('STRING', 'test')
    result = translator.xpath_contains_function(xpath, string_function)
    assert "contains(., 'test')" in result.post_condition
    
    # Test with IDENT argument
    ident_function = MockFunction('IDENT', 'test')
    result = translator.xpath_contains_function(xpath, ident_function)
    assert "contains(., 'test')" in result.post_condition
    
    # Test with invalid argument type
    invalid_function = MockFunction('NUMBER', '42')
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath, invalid_function)
    assert "Expected a single string or ident" in str(excinfo.value)


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    xpath = XPathExpr()
    
    # Mock function object with NUMBER argument type
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        
        class arguments:
            class value:
                def __init__(self, val):
                    self.value = val
                    
            arguments = [value(1)]
    
    mock_func = MockFunction()
    
    # Test case 1: lt(1) should add condition 'position() < 2'
    result = translator.xpath_lt_function(xpath, mock_func)
    assert result.post_condition == 'position() < 2'
    
    # Test case 2: Test with value 0
    mock_func2 = MockFunction()
    mock_func2.arguments[0].value = '0'
    xpath2 = XPathExpr()
    result2 = translator.xpath_lt_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() < 1'
    
    # Test case 3: Test error handling with wrong argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
    
    mock_func_invalid = MockFunctionInvalid()
    xpath3 = XPathExpr()
    
    try:
        translator.xpath_lt_function(xpath3, mock_func_invalid)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass  # Expected behavior
    
    # Test case 4: Verify the method returns the xpath object
    mock_func4 = MockFunction()
    mock_func4.arguments[0].value = '5'
    xpath4 = XPathExpr()
    result4 = translator.xpath_lt_function(xpath4, mock_func4)
    assert result4 is xpath4  # Should return the same object
    
    # Test case 5: Test with large number
    mock_func5 = MockFunction()
    mock_func5.arguments[0].value = '100'
    xpath5 = XPathExpr()
    result5 = translator.xpath_lt_function(xpath5, mock_func5)
    assert result5.post_condition == 'position() < 101'


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function object with NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with eq(0) - should add condition position() = 1
    function = MockFunction("0")
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == "position() = 1"
    
    # Test with eq(1) - should add condition position() = 2
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction("1")
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == "position() = 2"
    
    # Test with eq(5) - should add condition position() = 6
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction("5")
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == "position() = 6"
    
    # Test with non-NUMBER argument type should raise ExpressionError
    class MockFunctionInvalid:
        def __init__(self):
            self.arguments = ["invalid"]
        
        def argument_types(self):
            return ['STRING']
    
    xpath4 = translator.xpathexpr_cls()
    function4 = MockFunctionInvalid()
    try:
        translator.xpath_eq_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test raising ExpressionError for invalid argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath3, function3)


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments[0].value = '2'
    
    result = translator.xpath_lt_function(xpath, function)
    assert 'position() < 3' in str(result)
    
    # Test with edge case: value = 0
    xpath2 = translator.xpathexpr_cls()
    function2 = MagicMock()
    function2.argument_types.return_value = ['NUMBER']
    function2.arguments[0].value = '0'
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert 'position() < 1' in str(result2)
    
    # Test with negative value
    xpath3 = translator.xpathexpr_cls()
    function3 = MagicMock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments[0].value = '-1'
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert 'position() < 0' in str(result3)
    
    # Test that it raises ExpressionError for non-number arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = MagicMock()
    function4.argument_types.return_value = ['STRING']
    function4.arguments = ['not_a_number']
    
    try:
        translator.xpath_lt_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

```


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with a different index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '3'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 4'
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, function3)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with xpath object
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with different index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 3'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath3, function3)
```


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and function argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with index 5
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 6'
    
    # Test with negative index
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result4 = translator.xpath_eq_function(xpath4, function4)
    assert result4.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-number arguments
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), function5)
```


# LLM-generated content at query #35
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls(element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    assert result is xpath
    
    # Test with another number
    xpath2 = translator.xpathexpr_cls(element='h1')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 3'
    
    # Test with non-number argument type
    xpath3 = translator.xpathexpr_cls(element='h1')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_eq_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls(element='h1')
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    try:
        translator.xpath_eq_function(xpath4, function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'  # 2 + 1 = 3
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'  # -1 + 1 = 0
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'  # 0 + 1 = 1
    
    # Test that non-NUMBER argument raises ExpressionError
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with valid string argument
    xpath = XPathExpr()
    function = Mock(function_type='function', name='has', 
                    arguments=[Mock(value='.bar', type='STRING')])
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(' in str(result)
    
    # Test with valid ident argument
    xpath2 = XPathExpr()
    function2 = Mock(function_type='function', name='has',
                     arguments=[Mock(value='div', type='IDENT')])
    function2.argument_types = lambda: ['IDENT']
    result2 = translator.xpath_has_function(xpath2, function2)
    assert 'descendant::div' in str(result2)
    
    # Test with invalid argument type
    xpath3 = XPathExpr()
    function3 = Mock(function_type='function', name='has',
                     arguments=[Mock(value='1', type='NUMBER')])
    function3.argument_types = lambda: ['NUMBER']
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = XPathExpr()
    function4 = Mock(function_type='function', name='has',
                     arguments=[Mock(value='.foo', type='STRING'), 
                               Mock(value='.bar', type='STRING')])
    function4.argument_types = lambda: ['STRING', 'STRING']
    try:
        translator.xpath_has_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with no arguments
    xpath5 = XPathExpr()
    function5 = Mock(function_type='function', name='has',
                     arguments=[])
    function5.argument_types = lambda: []
    try:
        translator.xpath_has_function(xpath5, function5)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple case
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 1]'
    assert result.post_condition == 'position() > 1'
    
    # Test with value 2
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert str(result2) == '*[position() > 3]'
    assert result2.post_condition == 'position() > 3'
    
    # Test with existing condition
    xpath3 = translator.xpathexpr_cls()
    xpath3.add_condition("@class = 'test'")
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert str(result3) == '*[@class = \'test\'][position() > 1]'
    
    # Test with invalid argument type (should raise error)
    import pytest
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath4, function4)


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and number argument
    xpath = translator.xpathexpr_cls()
    class MockArgument:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '5'})()]
    
    result = translator.xpath_lt_function(xpath, MockArgument())
    assert result.post_condition == 'position() < 6'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    class MockArgumentZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '0'})()]
    
    result = translator.xpath_lt_function(xpath, MockArgumentZero())
    assert result.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    class MockArgumentNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '-1'})()]
    
    result = translator.xpath_lt_function(xpath, MockArgumentNegative())
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    class MockInvalidArgument:
        def argument_types(self):
            return ['STRING']
        arguments = [type('obj', (object,), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, MockInvalidArgument())
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    class MockMultipleArgs:
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
        arguments = [type('obj', (object,), {'value': '1'}),
                    type('obj', (object,), {'value': '2'})()]
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, MockMultipleArgs())


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('obj', (object,), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('obj', (object,), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)

    # Test with IDENT type
    xpath_ident = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('obj', (object,), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('obj', (object,), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath_ident)

    # Test error with invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div', element='div'),
            type('obj', (object,), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('obj', (object,), {'value': '1'})()]
            })()
        )

    # Test that post_condition is properly added
    xpath_test = translator.xpathexpr_cls(path='//div', element='div')
    result = translator.xpath_has_function(
        xpath_test,
        type('obj', (object,), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('obj', (object,), {'value': '.test'})()]
        })()
    )
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    assert 'test' in result.post_condition

    # Test with complex selector
    xpath_complex = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('obj', (object,), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('obj', (object,), {'value': 'span.class'})()]
        })()
    )
    assert 'descendant::' in str(xpath_complex)
    assert 'span' in str(xpath_complex)
    assert 'class' in str(xpath_complex)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector with class
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test has selector with element
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has selector with no matches
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.baz'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " baz ")]' in str(xpath)


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_first_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_first_pseudo(xpath)
    assert result.post_condition == 'position() = 1'
    assert result is xpath


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_image_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_image_pseudo(xpath)
    assert "@type = 'image' and name(.) = 'input'" in str(result)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Mock function with NUMBER argument
    class MockFunction:
        class Argument:
            def __init__(self, value):
                self.value = value
        
        def argument_types(self):
            return ['NUMBER']
        
        arguments = [Argument('2')]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    
    assert result.post_condition == 'position() > 3'
    assert result is xpath

def test_JQueryTranslator_xpath_gt_function_invalid_args():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    class MockFunction:
        class Argument:
            def __init__(self, value):
                self.value = value
        
        def argument_types(self):
            return ['STRING']
        
        arguments = [Argument('invalid')]
    
    try:
        translator.xpath_gt_function(xpath, MockFunction())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_password_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_password_pseudo(xpath)
    
    assert "@type = 'password' and name(.) = 'input'" in str(result)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        MagicMock(
            argument_types=lambda: ['STRING'],
            arguments=[MagicMock(value='title')]
        )
    )
    assert "contains(., 'title')" in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        MagicMock(
            argument_types=lambda: ['IDENT'],
            arguments=[MagicMock(value='text')]
        )
    )
    assert "contains(., 'text')" in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            MagicMock(
                argument_types=lambda: ['NUMBER'],
                arguments=[MagicMock(value='42')]
            )
        )
```


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test basic functionality
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, MagicMock(arguments=[MagicMock(value='0')], argument_types=lambda: ['NUMBER']))
    assert result.post_condition == 'position() > 1'
    
    # Test with different index values
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, MagicMock(arguments=[MagicMock(value='2')], argument_types=lambda: ['NUMBER']))
    assert result.post_condition == 'position() > 3'
    
    # Test with negative index
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, MagicMock(arguments=[MagicMock(value='-1')], argument_types=lambda: ['NUMBER']))
    assert result.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-number arguments
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), MagicMock(arguments=[MagicMock(value='string')], argument_types=lambda: ['STRING']))
    
    # Test that it returns the same xpath object
    xpath = translator.xpathexpr_cls()
    returned_xpath = translator.xpath_gt_function(xpath, MagicMock(arguments=[MagicMock(value='5')], argument_types=lambda: ['NUMBER']))
    assert returned_xpath is xpath
```


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test text'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert "contains(., 'test text')" in str(result)
    
    # Test with IDENT argument
    xpath2 = XPathExpr()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'text'})()]
    
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert "contains(., 'text')" in str(result2)
    
    # Test with invalid argument type raises ExpressionError
    xpath3 = XPathExpr()
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '42'})()]
    
    try:
        translator.xpath_contains_function(xpath3, MockFunctionInvalid())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with empty string
    xpath4 = XPathExpr()
    class MockFunctionEmpty:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': ''})()]
    
    result4 = translator.xpath_contains_function(xpath4, MockFunctionEmpty())
    assert "contains(., '')" in str(result4)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    from cssselect.xpath import XPathExpr
    xpath = XPathExpr()
    
    # Mock function with NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        class Argument:
            def __init__(self, value):
                self.value = value
        arguments = [Argument('1')]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 2'
    
    # Test with value 0
    xpath2 = XPathExpr()
    mock_func2 = MockFunction()
    mock_func2.arguments = [MockFunction.Argument('0')]
    result2 = translator.xpath_lt_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath3 = XPathExpr()
    mock_func3 = MockFunction()
    mock_func3.arguments = [MockFunction.Argument('-1')]
    result3 = translator.xpath_lt_function(xpath3, mock_func3)
    assert result3.post_condition == 'position() < 0'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = ['test']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(XPathExpr(), MockFunctionString())


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    xpath.element = 'h1'
    
    class MockNumber:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '0'})()]
    
    function = MockNumber()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with different index
    xpath2 = translator.xpathexpr_cls()
    xpath2.element = 'h1'
    function2 = MockNumber()
    function2.arguments[0].value = '2'
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 3'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    
    xpath3 = translator.xpathexpr_cls()
    try:
        translator.xpath_eq_function(xpath3, MockString())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '0'})()]
        })()
    )
    assert 'position() > 1' in str(xpath)
    
    # Test with different index value
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '2'})()]
        })()
    )
    assert 'position() > 3' in str(xpath)
    
    # Test with negative index
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '-1'})()]
        })()
    )
    assert 'position() > 0' in str(xpath)
    
    # Test that non-NUMBER argument raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['STRING'],
                'arguments': [type('Argument', (), {'value': 'test'})()]
            })()
        )
```


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = XPathExpr()
    result = translator.xpath_eq_function(xpath, create_mock_function('NUMBER', '0'))
    assert result.post_condition == 'position() = 1'
    
    # Test with different index values
    xpath2 = XPathExpr()
    result2 = translator.xpath_eq_function(xpath2, create_mock_function('NUMBER', '2'))
    assert result2.post_condition == 'position() = 3'
    
    # Test with negative index? Actually shouldn't happen but test edge case
    xpath3 = XPathExpr()
    result3 = translator.xpath_eq_function(xpath3, create_mock_function('NUMBER', '-1'))
    assert result3.post_condition == 'position() = 0'
    
    # Test that invalid argument type raises ExpressionError
    try:
        translator.xpath_eq_function(XPathExpr(), create_mock_function('STRING', 'test'))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that multiple arguments raises ExpressionError
    try:
        translator.xpath_eq_function(XPathExpr(), create_mock_function('NUMBER', '1', '2'))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

def create_mock_function(argument_type, *values):
    """Helper to create a mock function object for testing"""
    class MockFunction:
        def argument_types(self):
            return [argument_type] if len(values) == 1 else ['NUMBER', 'NUMBER']
        
        def __init__(self):
            self.arguments = [type('obj', (object,), {'value': v})() for v in values]
    
    return MockFunction()
```


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    from cssselect.xpath import XPathExpr
    xpath = XPathExpr(path='//div', element='div')
    
    # Create a mock function with NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
    
    # Test with value 0 (should match position < 1)
    function = MockFunction('0')
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with value 1 (should match position < 2)
    xpath2 = XPathExpr(path='//div', element='div')
    function2 = MockFunction('1')
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 2'
    
    # Test with value 5 (should match position < 6)
    xpath3 = XPathExpr(path='//div', element='div')
    function3 = MockFunction('5')
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 6'
    
    # Test that non-NUMBER arguments raise ExpressionError
    class MockStringFunction:
        def argument_types(self):
            return ['STRING']
        
        def __init__(self):
            self.arguments = []
    
    xpath4 = XPathExpr(path='//div', element='div')
    try:
        translator.xpath_lt_function(xpath4, MockStringFunction())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that invalid number raises ValueError
    xpath5 = XPathExpr(path='//div', element='div')
    function5 = MockFunction('not_a_number')
    try:
        translator.xpath_lt_function(xpath5, function5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with a different index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, function3)


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('STRING', '.bar'))
    assert "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]" in str(result)
    
    # Test has with element selector
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('IDENT', 'span'))
    assert "descendant::span" in str(result)
    
    # Test has with no matching elements
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('STRING', '.nonexistent'))
    assert "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' nonexistent ')]" in str(result)


class MockFunction:
    def __init__(self, arg_type, value):
        self.arguments = [MockArgument(arg_type, value)]
    
    def argument_types(self):
        return [self.arguments[0].type]


class MockArgument:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value
```


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '2'})()]
    })()
    result = translator.xpath_gt_function(xpath, mock_function)
    assert result.post_condition == 'position() > 3'
    
    # Test with 0 index (first element)
    xpath = translator.xpathexpr_cls()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, mock_function)
    assert result.post_condition == 'position() > 1'
    
    # Test that it raises ExpressionError for non-numeric arguments
    xpath = translator.xpathexpr_cls()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, mock_function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with a simple string argument
    result = translator.xpath_has_function(xpath, MockFunction("STRING", ".bar"))
    assert "descendant::" in result.post_condition
    assert result is xpath
    
    # Test with an IDENT argument
    xpath2 = XPathExpr()
    result2 = translator.xpath_has_function(xpath2, MockFunction("IDENT", "div"))
    assert "descendant::" in result2.post_condition
    assert result2 is xpath2
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    try:
        translator.xpath_has_function(XPathExpr(), MockFunction("NUMBER", "42"))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

class MockFunction:
    def __init__(self, arg_type, arg_value):
        self.arguments = [MockArgument(arg_type, arg_value)]
    
    def argument_types(self):
        return [self.arguments[0].type]

class MockArgument:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value
```


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Mock function with NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 3'
    
    # Test with negative index
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionNeg:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '-1'})()]
    
    result2 = translator.xpath_gt_function(xpath2, MockFunctionNeg())
    assert result2.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), MockFunctionInvalid())


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})]
        })()
    )
    assert 'descendant::*[contains(concat(\' \',@class,\' \'), \' bar \')]' in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})]
            })()
        )
```


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'
    
    # Test with different number
    xpath2 = XPathExpr()
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '0'})()]
    })()
    result2 = translator.xpath_lt_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number (Edge case)
    xpath3 = XPathExpr()
    mock_func3 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_lt_function(xpath3, mock_func3)
    assert result3.post_condition == 'position() < 0'


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = []
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, MockFunctionInvalid())


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with 0 index
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type (STRING)
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that xpath element is preserved
    xpath = XPathExpr(element='div', condition='@class')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.element == 'div'
    assert result.condition == '@class'
    assert result.post_condition == 'position() < 6'


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    from cssselect.parser import Function, parse
    # Create a function with STRING argument
    function = Function('contains', [parse('"title"')[0]])
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = Function('contains', [parse('title')[0]])
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert "contains(., 'title')" in str(result2)
    
    # Test that it raises ExpressionError for wrong argument types
    from cssselect.xpath import ExpressionError
    xpath3 = translator.xpathexpr_cls()
    function3 = Function('contains', [parse('123')[0]])
    try:
        translator.xpath_contains_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = Function('contains', [parse('"text"')[0], parse('"extra"')[0]])
    try:
        translator.xpath_contains_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with value 2
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative value
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test with non-NUMBER argument type
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    try:
        translator.xpath_gt_function(xpath4, function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [
            type('Argument', (), {'value': '1'}),
            type('Argument', (), {'value': '2'})
        ]
    })()
    
    try:
        translator.xpath_gt_function(xpath5, function5)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test case 1: Normal usage with valid number
    xpath = translator.xpathexpr_cls(path='//div/h1')
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert 'position() < 2' in str(result)
    
    # Test case 2: With value 0
    xpath2 = translator.xpathexpr_cls(path='//div/p')
    function2 = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '0'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert 'position() < 1' in str(result2)
    
    # Test case 3: With larger number
    xpath3 = translator.xpathexpr_cls(path='//ul/li')
    function3 = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '5'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert 'position() < 6' in str(result3)
    
    # Test case 4: Verify it preserves the original path
    xpath4 = translator.xpathexpr_cls(path='//div/section')
    function4 = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '2'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result4 = translator.xpath_lt_function(xpath4, function4)
    assert '//div/section' in str(result4)
    assert 'position() < 3' in str(result4)


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1', element='h1')
    
    # Test with number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath2 = translator.xpathexpr_cls(path='//h1', element='h1')
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '-1'})()]
    
    result2 = translator.xpath_gt_function(xpath2, MockFunctionNegative())
    assert result2.post_condition == 'position() > 0'


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'title'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert 'contains(., "title")' in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'text'})()]
    
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert 'contains(., "text")' in str(result2)


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with IDENT argument
    func = Function('contains', [parse('title')[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with invalid argument type
    from cssselect.parser import Number
    func = Function('contains', [Number('1')])
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls(), func)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    func = Function('contains', [parse('"text"')[0], parse('"more"')[0]])
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls(), func)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, Token
    from cssselect.xpath import XPathExpr
    
    # Create a mock function with number argument
    function = Function('eq', [Token('NUMBER', '0')])
    function.arguments = [Token('NUMBER', '0')]
    
    xpath = XPathExpr('div', 'div', '')
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "div[position() = 1]"
    
    # Test with negative number
    function.arguments = [Token('NUMBER', '-1')]
    xpath = XPathExpr('div', 'div', '')
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "div[position() = 0]"
    
    # Test with larger number
    function.arguments = [Token('NUMBER', '5')]
    xpath = XPathExpr('div', 'div', '')
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "div[position() = 6]"
    
    # Test that invalid argument types raise ExpressionError
    function.arguments = [Token('STRING', 'test')]
    try:
        translator.xpath_eq_function(XPathExpr('div', 'div', ''), function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'title'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    translator2 = JQueryTranslator()
    xpath2 = translator2.xpathexpr_cls()
    
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('arg', (), {'value': 'text'})()]
    
    result2 = translator2.xpath_contains_function(xpath2, MockFunctionIdent())
    assert "contains(., 'text')" in str(result2)
    
    # Test with invalid argument types
    translator3 = JQueryTranslator()
    xpath3 = translator3.xpathexpr_cls()
    
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '1'})()]
    
    try:
        translator3.xpath_contains_function(xpath3, MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test multiple post_conditions
    translator4 = JQueryTranslator()
    xpath4 = translator4.xpathexpr_cls()
    xpath4.add_post_condition('position() = 1')
    
    result4 = translator4.xpath_contains_function(xpath4, MockFunctionString())
    assert "position() = 1" in str(result4)
    assert "contains(., 'title')" in str(result4)


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = XPathExpr()
    function = MockFunction(['NUMBER'], [MockArgument('1')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'
    
    # Test with another valid number
    xpath = XPathExpr()
    function = MockFunction(['NUMBER'], [MockArgument('3')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 4'
    
    # Test with zero
    xpath = XPathExpr()
    function = MockFunction(['NUMBER'], [MockArgument('0')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with negative number
    xpath = XPathExpr()
    function = MockFunction(['NUMBER'], [MockArgument('-1')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-number arguments
    xpath = XPathExpr()
    function = MockFunction(['STRING'], [MockArgument('test')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for multiple arguments
    xpath = XPathExpr()
    function = MockFunction(['NUMBER', 'NUMBER'], [MockArgument('1'), MockArgument('2')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test has with no matching elements
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.baz'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " baz ")]' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'container'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " container ")]' in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with string argument
    from cssselect.parser import Function, Token
    func = Function('contains', [Token('STRING', 'test')])
    result = translator.xpath_contains_function(xpath, func)
    assert result.post_condition == "contains(., 'test')"
    
    # Test with ident argument
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [Token('IDENT', 'title')])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert result2.post_condition == "contains(., 'title')"
    
    # Test that invalid argument types raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    func3 = Function('contains', [Token('NUMBER', '42')])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='@class'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert xpath.post_condition is not None
    assert 'descendant::' in xpath.post_condition
    
    # Test with IDENT type
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert xpath.post_condition is not None
    assert 'descendant::' in xpath.post_condition
```


# LLM-generated content at query #35
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    # Create a mock function with STRING argument
    func = Function('contains', [parse('"title"')])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with IDENT argument (no quotes)
    func2 = Function('contains', [parse('title')])
    xpath2 = translator.xpath_contains_function(translator.xpathexpr_cls(), func2)
    assert 'contains(., "title")' in str(xpath2)
    
    # Test that it raises ExpressionError for invalid argument types
    from cssselect.parser import parse as css_parse
    func3 = Function('contains', [css_parse('1')])  # NUMBER type
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Mock XPathExpr object
    class MockXPathExpr:
        def __init__(self):
            self.post_condition = None
            
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    # Mock function object
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
            
        def argument_types(self):
            return ['NUMBER']
    
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    # Test basic case
    xpath = MockXPathExpr()
    function = MockFunction("0")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with positive index
    xpath = MockXPathExpr()
    function = MockFunction("2")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with negative index
    xpath = MockXPathExpr()
    function = MockFunction("-1")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test error case with non-numeric argument
    class MockStringFunction:
        def __init__(self):
            self.arguments = ["string"]
            
        def argument_types(self):
            return ['STRING']
    
    xpath = MockXPathExpr()
    function = MockStringFunction()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that xpath is returned
    xpath = MockXPathExpr()
    function = MockFunction("5")
    assert translator.xpath_gt_function(xpath, function) is xpath


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function object for :eq(0)
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '0'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test :eq(1)
    xpath2 = translator.xpathexpr_cls()
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('obj', (object,), {'value': '1'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('obj', (object,), {'value': 'test'})()]
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(translator.xpathexpr_cls(), InvalidFunction())
```


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    xpath = translator.xpathexpr_cls()
    func = Function('contains', [parse('"title"')[0]])
    result = translator.xpath_contains_function(xpath, func)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with ident argument
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('title')[0]])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert result2.post_condition == "contains(., 'title')"
    
    # Test that function returns the xpath object
    assert result is xpath
    
    # Test with invalid argument types
    from cssselect.parser import Function
    func3 = Function('contains', [parse('1')[0]])
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument containing "title"
    from cssselect.parser import Function, Token
    func = Function('contains', [Token('STRING', '"title"')])
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//h1', element='h1'), 
        func
    )
    assert str(xpath) == "//h1[contains(., 'title')]"
    
    # Test with IDENT argument
    func2 = Function('contains', [Token('IDENT', 'hello')])
    xpath2 = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//p', element='p'), 
        func2
    )
    assert str(xpath2) == "//p[contains(., 'hello')]"
    
    # Test with invalid argument type (should raise ExpressionError)
    from cssselect.xpath import ExpressionError
    import pytest
    func3 = Function('contains', [Token('NUMBER', '42')])
    with pytest.raises(ExpressionError, match="Expected a single string or ident"):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(path='//div', element='div'), 
            func3
        )


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test with ident type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test invalid argument type raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '42'})()]
            })()
        )
```


