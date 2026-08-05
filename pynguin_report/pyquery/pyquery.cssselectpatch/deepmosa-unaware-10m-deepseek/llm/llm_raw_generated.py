####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'title'})()]
        })()
    )
    assert 'contains(., "title")' in str(xpath)
    assert 'h1' in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'content'})()]
        })()
    )
    assert 'contains(., "content")' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test invalid argument type raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '123'})()]
            })()
        )


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    from cssselect.parser import Function, parse
    function = Function('contains', [parse('"title"')[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), function)
    assert "contains(., 'title')" in str(xpath)
    
    # Test with IDENT argument type
    function = Function('contains', [parse('title')[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), function)
    assert "contains(., 'title')" in str(xpath)
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    function = Function('contains', [parse('123')[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(translator.xpathexpr_cls(), function)


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_hidden_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpath_hidden_pseudo(XPathExpr())
    assert "@type = 'hidden' and name(.) = 'input'" in str(xpath)


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0 (first element)
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test with invalid argument type (should raise ExpressionError)
    import pytest
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, function4)
```


# LLM-generated content at query #5
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
        def __init__(self, args):
            self.arguments = args
            
        def argument_types(self):
            return ['NUMBER']
    
    # Test with eq(0) - should match first element (position() = 1)
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with eq(1) - should match second element (position() = 2)
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction([MockArgument('1')])
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with eq(5) - should match sixth element (position() = 6)
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction([MockArgument('5')])
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 6'
    
    # Test that non-NUMBER argument types raise ExpressionError
    class MockStringFunction:
        def __init__(self, args):
            self.arguments = args
            
        def argument_types(self):
            return ['STRING']
    
    xpath4 = translator.xpathexpr_cls()
    function4 = MockStringFunction([MockArgument('test')])
    try:
        translator.xpath_eq_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that multiple arguments raise ExpressionError
    class MockMultipleFunction:
        def __init__(self, args):
            self.arguments = args
            
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
    
    xpath5 = translator.xpathexpr_cls()
    function5 = MockMultipleFunction([MockArgument('1'), MockArgument('2')])
    try:
        translator.xpath_eq_function(xpath5, function5)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic has selector matching descendant
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test 2: Has with IDENT type
    xpath2 = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath2)
    assert 'div' in str(xpath2)
    
    # Test 3: Raises ExpressionError for invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )
    
    # Test 4: Post condition is properly added
    xpath3 = translator.xpathexpr_cls()
    result = translator.xpath_has_function(
        xpath3,
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.test'})()]
        })()
    )
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
```


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Create a mock XPath expression
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function with a NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should give position() < 1)
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with value 1 (should give position() < 2)
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction([MockArgument('1')])
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 2'
    
    # Test with value 5 (should give position() < 6)
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction([MockArgument('5')])
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 6'
    
    # Test with negative value (-1 should give position() < 0)
    xpath4 = translator.xpathexpr_cls()
    function4 = MockFunction([MockArgument('-1')])
    result4 = translator.xpath_lt_function(xpath4, function4)
    assert result4.post_condition == 'position() < 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class WrongFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['STRING']
    
    xpath5 = translator.xpathexpr_cls()
    wrong_function = WrongFunction([MockArgument('test')])
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath5, wrong_function)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = XPathExpr(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with IDENT argument
    xpath2 = XPathExpr(path='//div', element='div')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition is not None
    assert 'descendant::' in result2.post_condition
    
    # Test that it raises ExpressionError for invalid argument types
    xpath3 = XPathExpr(path='//div', element='div')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '5'})()]
    })()
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for empty arguments
    xpath4 = XPathExpr(path='//div', element='div')
    function4 = type('Function', (), {
        'argument_types': lambda self: [],
        'arguments': []
    })()
    try:
        translator.xpath_has_function(xpath4, function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Verify the post_condition contains the correct CSS selector converted to XPath
    xpath5 = XPathExpr(path='//div', element='div')
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.my-class'})()]
    })()
    result5 = translator.xpath_has_function(xpath5, function5)
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " my-class ")]' in result5.post_condition
```


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test case 1: has with string argument - element contains another element matching selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test case 2: has with ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test case 3: has with string argument - element does not contain matching elements
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition="@class='foo'"),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.baz'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " baz ")]' in str(xpath)
    
    # Test case 4: has with invalid argument type
    try:
        translator.xpath_has_function(
            XPathExpr(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '0'})()]
            })()
        )
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.arguments = [type('Arg', (), {'value': 'title'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.arguments = [type('Arg', (), {'value': 'text'})()]
    function.argument_types = lambda: ['IDENT']
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.arguments = [type('Arg', (), {'value': '1'})()]
    function.argument_types = lambda: ['NUMBER']
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.arguments = [type('Arg', (), {'value': 'a'}), type('Arg', (), {'value': 'b'})()]
    function.argument_types = lambda: ['STRING', 'STRING']
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with special characters in text
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.arguments = [type('Arg', (), {'value': "it's"})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., \"it's\")"


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    func = Function('has', [parse('".bar"')])
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        func
    )
    assert 'descendant' in str(xpath)
    
    # Test with ident argument
    func = Function('has', [parse('div')])
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        func
    )
    assert 'descendant' in str(xpath)
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    
    func = Function('has', [parse('123')])  # NUMBER type
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            func
        )


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    mock_xpath = XPathExpr()
    result = translator.xpath_eq_function(mock_xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    mock_xpath_zero = XPathExpr()
    result_zero = translator.xpath_eq_function(mock_xpath_zero, MockFunctionZero())
    assert result_zero.post_condition == 'position() = 1'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), MockFunctionInvalid())
    
    # Test that it raises ExpressionError for empty arguments
    class MockFunctionEmpty:
        def argument_types(self):
            return []
        arguments = []
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), MockFunctionEmpty())


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic has with class selector
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('.bar'))
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(result)
    
    # Test 2: Has with tag selector
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('span'))
    assert 'descendant::span' in str(result)
    
    # Test 3: Multiple conditions
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('.foo'))
    assert 'descendant::*[contains(concat(" ", @class, " "), " foo ")]' in str(result)
    
    # Test 4: Verify post_condition is added
    xpath = XPathExpr(path='//div', element='div')
    result = translator.xpath_has_function(xpath, MockFunction('.test'))
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test 5: Error on non-string/non-ident argument
    xpath = XPathExpr(path='//div', element='div')
    try:
        translator.xpath_has_function(xpath, MockFunction(function_type='NUMBER'))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


class MockFunction:
    """Helper class to mock the function argument for testing."""
    def __init__(self, value=None, function_type='STRING'):
        self.arguments = []
        if value is not None:
            self.arguments = [MockArgument(value)]
        self._type = function_type
    
    def argument_types(self):
        return [self._type]


class MockArgument:
    """Helper class to mock a single function argument."""
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    function = parse(":contains('test')")[0].parsed_selectors[0].pseudo_class
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'test')" in str(result)
    
    # Test with ident argument
    xpath2 = translator.xpathexpr_cls()
    function2 = parse(":contains(test)")[0].parsed_selectors[0].pseudo_class
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert "contains(., 'test')" in str(result2)
    
    # Test with invalid argument type
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = parse(":contains(123)")[0].parsed_selectors[0].pseudo_class
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, function3)


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock()]
    function.arguments[0].value = '2'
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function.arguments[0].value = '-1'
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    function.arguments[0].value = '0'
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with large number
    xpath = translator.xpathexpr_cls()
    function.arguments[0].value = '100'
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 101'
    
    # Test that it raises ExpressionError for non-number arguments
    xpath = translator.xpathexpr_cls()
    function.argument_types.return_value = ['STRING']
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it returns the same xpath object
    xpath = translator.xpathexpr_cls()
    function.argument_types.return_value = ['NUMBER']
    result = translator.xpath_gt_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Create a mock xpath object
    class MockXPath:
        def __init__(self):
            self.post_conditions = []
        
        def add_post_condition(self, condition):
            self.post_conditions.append(condition)
    
    xpath = MockXPath()
    
    # Test case 1: gt(0) should add condition position() > 1
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_conditions == ['position() > 1']
    
    # Test case 2: gt(5) should add condition position() > 6
    xpath2 = MockXPath()
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '5'})()]
    
    result2 = translator.xpath_gt_function(xpath2, MockFunction2())
    assert result2.post_conditions == ['position() > 6']
    
    # Test case 3: Invalid argument type should raise ExpressionError
    class MockFunction3:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(MockXPath(), MockFunction3())
```


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    func_mock = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, func_mock)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    func_mock2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, func_mock2)
    assert "contains(., 'test')" in str(result2)
    
    # Test that it raises ExpressionError for invalid argument types
    xpath3 = translator.xpathexpr_cls()
    func_mock3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath3, func_mock3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '0'})],
        'argument_types': lambda self: ['NUMBER'],
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})],
        'argument_types': lambda self: ['NUMBER'],
    })()
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with negative index should raise error
    import pytest
    with pytest.raises(ExpressionError):
        xpath3 = translator.xpathexpr_cls()
        function3 = type('Function', (), {
            'arguments': [type('Arg', (), {'value': 'not_a_number'})],
            'argument_types': lambda self: ['STRING'],
        })()
        translator.xpath_eq_function(xpath3, function3)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic string argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Argument', (), {'value': '.bar'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, MockFunctionString())
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'
    
    # Test 2: Ident argument
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('Argument', (), {'value': 'div'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, MockFunctionIdent())
    assert result.post_condition == 'descendant::div'
    
    # Test 3: Invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Argument', (), {'value': '1'})()]
    
    xpath = translator.xpathexpr_cls()
    try:
        translator.xpath_has_function(xpath, MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test 4: Multiple arguments
    class MockFunctionMultiple:
        def argument_types(self):
            return ['STRING', 'STRING']
        arguments = [type('Argument', (), {'value': '.foo'}), type('Argument', (), {'value': '.bar'})]
    
    xpath = translator.xpathexpr_cls()
    try:
        translator.xpath_has_function(xpath, MockFunctionMultiple())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test 5: Empty argument
    class MockFunctionEmpty:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Argument', (), {'value': ''})]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, MockFunctionEmpty())
    assert result.post_condition == 'descendant::*'
    
    # Test 6: Complex CSS selector
    class MockFunctionComplex:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Argument', (), {'value': 'div.foo > p.bar'})]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, MockFunctionComplex())
    assert 'descendant::' in result.post_condition
    assert 'div' in result.post_condition
    assert 'p' in result.post_condition
    
    # Test 7: Chain with existing post_condition
    xpath = translator.xpathexpr_cls()
    xpath.add_post_condition('position() = 1')
    result = translator.xpath_has_function(xpath, MockFunctionString())
    assert 'position() = 1' in result.post_condition
    assert 'descendant::' in result.post_condition
    
    # Test 8: Verify returns the same xpath object
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, MockFunctionString())
    assert result is xpath
    
    # Test 9: Verify post_condition is set correctly
    xpath = translator.xpathexpr_cls()
    xpath.add_post_condition('condition1')
    result = translator.xpath_has_function(xpath, MockFunctionString())
    assert 'condition1 and' in result.post_condition
    assert 'descendant::' in result.post_condition
    
    # Test 10: Verify the condition is combined with 'and' when multiple conditions exist
    xpath = translator.xpathexpr_cls()
    xpath.add_post_condition('condition1')
    result = translator.xpath_has_function(xpath, MockFunctionString())
    assert 'condition1 and (descendant::' in result.post_condition
```


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test basic functionality - eq(0) should match first element
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test eq(1) should match second element
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='1')]
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 2'
    
    # Test with negative number
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-number arguments
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)
    
    # Test that it works with existing post_condition
    xpath = XPathExpr()
    xpath.post_condition = 'existing_condition'
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'existing_condition and (position() = 3)'
```


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function with NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = []
    
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    # Test with value 0 (should become position() = 1)
    func = MockFunction()
    func.arguments = [MockArgument('0')]
    result = translator.xpath_eq_function(xpath, func)
    assert 'position() = 1' in str(result)
    
    # Test with value 1 (should become position() = 2)
    xpath2 = translator.xpathexpr_cls()
    func2 = MockFunction()
    func2.arguments = [MockArgument('1')]
    result2 = translator.xpath_eq_function(xpath2, func2)
    assert 'position() = 2' in str(result2)
    
    # Test with value 5 (should become position() = 6)
    xpath3 = translator.xpathexpr_cls()
    func3 = MockFunction()
    func3.arguments = [MockArgument('5')]
    result3 = translator.xpath_eq_function(xpath3, func3)
    assert 'position() = 6' in str(result3)
    
    # Test that non-NUMBER argument raises ExpressionError
    class MockStringFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [MockArgument('test')]
    
    xpath4 = translator.xpathexpr_cls()
    func4 = MockStringFunction()
    try:
        translator.xpath_eq_function(xpath4, func4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'content'})()]
    })()
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test with invalid argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains"):
        translator.xpath_contains_function(xpath3, function3)


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with number argument
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '0'})()]
        })()
    )
    assert 'position() > 1' in str(xpath)
    
    # Test with negative index
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '-1'})()]
        })()
    )
    assert 'position() > 0' in str(xpath)
    
    # Test raises ExpressionError for non-number argument
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(
            translator.xpathexpr_cls(element='h1'),
            type('Function', (), {
                'argument_types': lambda self: ['STRING'],
                'arguments': [type('Argument', (), {'value': 'test'})()]
            })()
        )
```


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test basic case: gt(0) should match elements at position > 1
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction('0'))
    assert 'position() > 1' in str(result)
    
    # Test with different index: gt(2) should match elements at position > 3
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction('2'))
    assert 'position() > 3' in str(result)
    
    # Test with negative index: gt(-1) should match elements at position > 0
    xpath = XPathExpr()
    result = translator.xpath_gt_function(xpath, MockFunction('-1'))
    assert 'position() > 0' in str(result)
    
    # Test that it raises ExpressionError for non-numeric arguments
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), MockFunction('string'))
    
    # Test that it raises ExpressionError for multiple arguments
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), MockFunction('1, 2'))


class MockFunction:
    """Helper class to mock the function argument"""
    def __init__(self, arg_value):
        self.arguments = [MockArgument(arg_value)]
    
    def argument_types(self):
        return ['NUMBER']


class MockArgument:
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with another valid number
    xpath = translator.xpathexpr_cls()
    function.arguments = [MagicMock(value='3')]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 4'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='invalid')]
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with different index value
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '3'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 4'
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :eq()"):
        translator.xpath_eq_function(xpath3, function3)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '3'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, mock_function)
    assert result.post_condition == 'position() < 4'
    
    # Test with another valid number
    xpath2 = translator.xpathexpr_cls()
    mock_function2 = type('MockFunction', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, mock_function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number? Actually CSS selector indices are non-negative, but test anyway
    xpath3 = translator.xpathexpr_cls()
    mock_function3 = type('MockFunction', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_lt_function(xpath3, mock_function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test error case with wrong argument types
    from cssselect.xpath import ExpressionError
    xpath4 = translator.xpathexpr_cls()
    mock_function4 = type('MockFunction', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('MockArg', (), {'value': 'test'})()]
    })()
    
    try:
        translator.xpath_lt_function(xpath4, mock_function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it returns the xpath object
    assert result is xpath
```


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})]
    })()
    
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert 'descendant-or-self::*' in result.post_condition or 'descendant::' in result.post_condition
    assert 'bar' in result.post_condition
    
    # Test with ident argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})]
    })()
    
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2 is xpath2
    assert 'descendant::' in result2.post_condition
    assert 'div' in result2.post_condition
    
    # Test invalid argument type
    import pytest
    from cssselect.xpath import ExpressionError
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 1})]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath3, function3)


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Create a mock XPath expression
    xpath = XPathExpr(path='//div', element='div')
    
    # Create a mock function with NUMBER argument type
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (first element)
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with value 1 (second element)
    xpath = XPathExpr(path='//div', element='div')
    function = MockFunction([MockArgument('1')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 2'
    
    # Test with value 5 (sixth element)
    xpath = XPathExpr(path='//div', element='div')
    function = MockFunction([MockArgument('5')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 6'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockNonNumberFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['STRING']
    
    xpath = XPathExpr(path='//div', element='div')
    function = MockNonNumberFunction([MockArgument('test')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for multiple arguments
    class MockMultipleArgsFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
    
    xpath = XPathExpr(path='//div', element='div')
    function = MockMultipleArgsFunction([MockArgument('1'), MockArgument('2')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is properly added when there's an existing post_condition
    xpath = XPathExpr(path='//div', element='div')
    xpath.post_condition = 'position() > 1'
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() > 1 and (position() = 1)'
```


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Create a mock XPath object
    class MockXPath:
        def __init__(self):
            self.post_condition = None
        
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    xpath = MockXPath()
    
    # Create a mock Function object with NUMBER argument
    class MockFunction:
        def __init__(self, value):
            self.arguments = [type('Arg', (), {'value': str(value)})()]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should become position() = 1)
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() = 1'
    
    # Test with value 1 (should become position() = 2)
    xpath2 = MockXPath()
    result2 = translator.xpath_eq_function(xpath2, MockFunction(1))
    assert result2.post_condition == 'position() = 2'
    
    # Test with value 5 (should become position() = 6)
    xpath3 = MockXPath()
    result3 = translator.xpath_eq_function(xpath3, MockFunction(5))
    assert result3.post_condition == 'position() = 6'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockFunctionInvalid:
        def __init__(self):
            self.arguments = ['invalid']
        
        def argument_types(self):
            return ['STRING']
    
    try:
        translator.xpath_eq_function(MockXPath(), MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for multiple arguments
    class MockFunctionMultiple:
        def __init__(self):
            self.arguments = [type('Arg', (), {'value': '1'})(), type('Arg', (), {'value': '2'})()]
        
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
    
    try:
        translator.xpath_eq_function(MockXPath(), MockFunctionMultiple())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_eq_function(xpath2, MockFunctionZero())
    assert result2.post_condition == 'position() = 1'
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(translator.xpathexpr_cls(), MockFunctionInvalid())
```


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic functionality - matching descendant
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test with no matching elements
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.nonexistent'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " nonexistent ")]' in str(xpath)
    
    # Test with tag selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert 'descendant::span' in str(xpath)
    
    # Test error handling for invalid arguments
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid input
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('gt', [parse('1')])
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, func)
    assert str(result) == '*[position() > 2]'
    
    # Test with zero index
    func = Function('gt', [parse('0')])
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, func)
    assert str(result) == '*[position() > 1]'
    
    # Test with negative number
    func = Function('gt', [parse('-1')])
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, func)
    assert str(result) == '*[position() > 0]'
    
    # Test with non-number argument should raise ExpressionError
    from cssselect.xpath import ExpressionError
    try:
        func = Function('gt', [parse('"string"')])
        xpath = translator.xpathexpr_cls()
        translator.xpath_gt_function(xpath, func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments should raise ExpressionError
    try:
        func = Function('gt', [parse('1'), parse('2')])
        xpath = translator.xpathexpr_cls()
        translator.xpath_gt_function(xpath, func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test chaining with other conditions
    xpath = translator.xpathexpr_cls(element='div', condition='@class')
    func = Function('gt', [parse('2')])
    result = translator.xpath_gt_function(xpath, func)
    assert str(result) == 'div[@class][position() > 3]'
```


# LLM-generated content at query #35
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
    
    assert result.post_condition == 'position() < 3'  # 2 + 1 = 3
    
    # Test with position 0
    xpath2 = translator.xpathexpr_cls()
    function2 = MagicMock()
    function2.argument_types.return_value = ['NUMBER']
    function2.arguments = [MagicMock(value='0')]
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    
    assert result2.post_condition == 'position() < 1'  # 0 + 1 = 1
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    function3 = MagicMock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [MagicMock(value='-1')]
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    
    assert result3.post_condition == 'position() < 0'  # -1 + 1 = 0
    
    # Test that it raises ExpressionError for non-number arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = MagicMock()
    function4.argument_types.return_value = ['STRING']
    function4.arguments = [MagicMock(value='test')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, function4)


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_expr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock()]
    function.arguments[0].value = 'title'
    
    result = translator.xpath_contains_function(xpath, function)
    assert 'contains(., "title")' in str(result)
    assert result.post_condition == 'contains(., "title")'
    
    # Test with IDENT argument
    xpath2 = translator.xpath_expr_cls()
    function2 = Mock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments = [Mock()]
    function2.arguments[0].value = 'hello'
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert 'contains(., "hello")' in str(result2)
    assert result2.post_condition == 'contains(., "hello")'
    
    # Test with invalid argument type
    xpath3 = translator.xpath_expr_cls()
    function3 = Mock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [Mock()]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, function3)


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1 (should add 1 to get position 2)
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test that non-NUMBER argument type raises ExpressionError
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :eq"):
        translator.xpath_eq_function(xpath3, function3)
```


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(result)
    assert result.post_condition is not None
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::div' in str(result)
    
    # Test with no match
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", @class, " "), " baz ")]' in str(result)
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('eq', [parse('0')[0]])
    xpath = translator.xpath_eq_function(
        translator.xpathexpr_cls(path='//h1'), func
    )
    assert str(xpath) == '//h1[position() = 1]'
    
    # Test with different number
    func = Function('eq', [parse('5')[0]])
    xpath = translator.xpath_eq_function(
        translator.xpathexpr_cls(path='//div'), func
    )
    assert str(xpath) == '//div[position() = 6]'
    
    # Test with negative number
    func = Function('eq', [parse('-1')[0]])
    xpath = translator.xpath_eq_function(
        translator.xpathexpr_cls(path='//p'), func
    )
    assert str(xpath) == '//p[position() = 0]'
    
    # Test that non-number argument raises ExpressionError
    try:
        from cssselect.parser import StringVal
        func = Function('eq', [StringVal('invalid')])
        translator.xpath_eq_function(
            translator.xpathexpr_cls(path='//h1'), func
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer" in str(e)
```


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    import pytest
    with pytest.raises(ExpressionError):
        invalid_function = type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
        translator.xpath_lt_function(translator.xpathexpr_cls(), invalid_function)


# LLM-generated content at query #41
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 3'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    mock_func2 = MockFunction()
    mock_func2.arguments[0].value = '0'
    result2 = translator.xpath_gt_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() > 1'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    mock_func3 = MockFunction()
    mock_func3.arguments[0].value = '-1'
    result3 = translator.xpath_gt_function(xpath3, mock_func3)
    assert result3.post_condition == 'position() > 0'
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = ['test']
    
    try:
        translator.xpath_gt_function(xpath, InvalidFunction())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    class MultipleArgsFunction:
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
        arguments = [type('MockArgument', (), {'value': '1'})(), type('MockArgument', (), {'value': '2'})()]
    
    try:
        translator.xpath_gt_function(xpath, MultipleArgsFunction())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #42
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Create a mock function with STRING argument type
    class MockArgument:
        def __init__(self, value, type='STRING'):
            self.value = value
            self.type = type
    
    class MockFunction:
        def __init__(self, arguments):
            self.arguments = arguments
        
        def argument_types(self):
            return [arg.type for arg in self.arguments]
    
    # Test with string argument
    mock_xpath = XPathExpr()
    mock_arg = MockArgument("test text", 'STRING')
    mock_function = MockFunction([mock_arg])
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert "contains(., 'test text')" in str(result)
    
    # Test with IDENT argument
    mock_xpath2 = XPathExpr()
    mock_arg2 = MockArgument("title", 'IDENT')
    mock_function2 = MockFunction([mock_arg2])
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function2)
    assert "contains(., 'title')" in str(result2)
    
    # Test with invalid argument type
    mock_xpath3 = XPathExpr()
    mock_arg3 = MockArgument("1", 'NUMBER')
    mock_function3 = MockFunction([mock_arg3])
    try:
        translator.xpath_contains_function(mock_xpath3, mock_function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident" in str(e)
```


# LLM-generated content at query #43
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple case
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #44
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '0'})()],
            'name': 'gt'
        })()
    )
    assert str(xpath) == "//div[position() > 1]"
    
    # Test with different index value
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '2'})()],
            'name': 'gt'
        })()
    )
    assert str(xpath) == "//div[position() > 3]"
    
    # Test that it raises ExpressionError for non-NUMBER argument
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(
            translator.xpathexpr_cls(path='//div'),
            type('Function', (), {
                'argument_types': lambda self: ['STRING'],
                'arguments': [type('Argument', (), {'value': 'test'})()],
                'name': 'gt'
            })()
        )
```


# LLM-generated content at query #45
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = XPathExpr(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::' in str(result)
    assert 'bar' in str(result)
    
    # Test with IDENT argument type
    xpath = XPathExpr(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::' in str(result)
    assert 'div' in str(result)
    
    # Test that post_condition is added correctly
    xpath = XPathExpr(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.test'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    assert 'test' in result.post_condition
    
    # Test with invalid argument type
    xpath = XPathExpr(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that the returned xpath has the original path preserved
    xpath = XPathExpr(path='//div[@class="foo"]', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert '//div[@class="foo"]' in str(result) or '//div[@class="foo"]' in result.path
```


# LLM-generated content at query #46
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic :has() with class selector
    xpath = translator.css_to_xpath('.foo:has(".bar")')
    assert 'descendant::' in xpath
    assert '@class' in xpath or 'contains' in xpath
    
    # Test :has() with element selector
    xpath = translator.css_to_xpath('.foo:has(div)')
    assert 'descendant::div' in xpath
    
    # Test :has() with no match
    xpath = translator.css_to_xpath('.foo:has(".baz")')
    assert 'descendant::' in xpath
    assert '@class' in xpath or 'contains' in xpath
    
    # Test :has() raises ExpressionError for invalid arguments
    from cssselect.parser import Function
    from cssselect.xpath import ExpressionError
    
    xpath_obj = translator.xpathexpr_cls()
    function = Function('has', ['1'])  # Invalid: number instead of string
    function.argument_types = lambda: ['NUMBER']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath_obj, function)


# LLM-generated content at query #47
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test with ident argument  
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument types
    from cssselect.xpath import ExpressionError
    import pytest
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #48
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test 1: Normal case - lt(1) should match first element (position < 2)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 2]'
    
    # Test 2: lt(0) should match no elements (position < 1)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 1]'
    
    # Test 3: Invalid argument type should raise ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test 4: Multiple arguments should raise ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'}), type('Argument', (), {'value': '2'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test 5: Negative index
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 0]'  # position < 0, which matches nothing
    
    # Test 6: Large index
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '100'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 101]'  # position < 101
    
    # Test 7: Verify post_condition is properly set (not condition)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 6'
```


# LLM-generated content at query #49
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArg', (), {'value': '2'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    
    assert result.post_condition == 'position() = 3'
    assert result is xpath


# LLM-generated content at query #50
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test valid eq function with number argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    # Create a simple xpath to test with
    xpath = XPathExpr('div', 'div', '')
    
    # Create a mock function with NUMBER argument
    class MockArgument:
        def __init__(self, value, type):
            self.value = value
            self.type = type
            
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(str(value), 'NUMBER')]
            
        def argument_types(self):
            return ['NUMBER']
    
    # Test eq(0) - first element (position() = 1)
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() = 1'
    
    # Test eq(1) - second element (position() = 2)
    result = translator.xpath_eq_function(XPathExpr('div', 'div', ''), MockFunction(1))
    assert result.post_condition == 'position() = 2'
    
    # Test eq(5) - sixth element (position() = 6)
    result = translator.xpath_eq_function(XPathExpr('div', 'div', ''), MockFunction(5))
    assert result.post_condition == 'position() = 6'
    
    # Test that invalid argument type raises ExpressionError
    class InvalidFunction:
        def __init__(self):
            self.arguments = [MockArgument('foo', 'STRING')]
            
        def argument_types(self):
            return ['STRING']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr('div', 'div', ''), InvalidFunction())
    
    # Test that multiple arguments raise ExpressionError
    class MultipleArgsFunction:
        def __init__(self):
            self.arguments = [MockArgument('1', 'NUMBER'), MockArgument('2', 'NUMBER')]
            
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr('div', 'div', ''), MultipleArgsFunction())
```


# LLM-generated content at query #51
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    class MockFunction:
        class Argument:
            def __init__(self, value):
                self.value = value
        
        def argument_types(self):
            return ['NUMBER']
        
        def __init__(self, arguments):
            self.arguments = arguments
    
    class MockXPath:
        def __init__(self):
            self.post_condition = None
        
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    # Test with value 0 (first element)
    xpath = MockXPath()
    function = MockFunction([MockFunction.Argument('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with value 1 (second element)
    xpath = MockXPath()
    function = MockFunction([MockFunction.Argument('1')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 2'
    
    # Test with value 5 (sixth element)
    xpath = MockXPath()
    function = MockFunction([MockFunction.Argument('5')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 6'
    
    # Test with negative value
    xpath = MockXPath()
    function = MockFunction([MockFunction.Argument('-1')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        
        def __init__(self):
            self.arguments = [MockFunction.Argument('text')]
    
    xpath = MockXPath()
    function = InvalidFunction()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    class MultipleArgsFunction:
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
        
        def __init__(self):
            self.arguments = [MockFunction.Argument('1'), MockFunction.Argument('2')]
    
    xpath = MockXPath()
    function = MultipleArgsFunction()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #52
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, Number
    func = Function('lt', [Number('1')])
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == 'self::*[position() < 2]'
    
    # Test with different number
    func = Function('lt', [Number('0')])
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == 'self::*[position() < 1]'
    
    # Test with element type
    func = Function('lt', [Number('2')])
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(element='div'), func)
    assert str(xpath) == 'self::div[position() < 3]'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    import pytest
    from cssselect.parser import StringVal
    func = Function('lt', [StringVal('"test"')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(translator.xpathexpr_cls(), func)
```


# LLM-generated content at query #53
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1', element='h1')
    
    # Test with STRING argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == "//h1[contains(., 'title')]"
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls(path='//div', element='div')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'content'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert str(result2) == "//div[contains(., 'content')]"
    
    # Test with empty string
    xpath3 = translator.xpathexpr_cls(path='//p', element='p')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': ''})()]
    })()
    result3 = translator.xpath_contains_function(xpath3, function3)
    assert str(result3) == "//p[contains(., '')]"
    
    # Test with special characters in string
    xpath4 = translator.xpathexpr_cls(path='//span', element='span')
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': "it's"})()]
    })()
    result4 = translator.xpath_contains_function(xpath4, function4)
    assert "it's" in str(result4)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    xpath5 = translator.xpathexpr_cls(path='//a', element='a')
    function5 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '42'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath5, function5)
```


# LLM-generated content at query #54
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, mock_function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    mock_function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, mock_function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test error case with wrong argument type
    xpath3 = translator.xpathexpr_cls()
    mock_function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_lt_function(xpath3, mock_function3)
```


# LLM-generated content at query #55
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic contains with string
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(element='h1'),
        MockFunction(['STRING'], ['title'])
    )
    assert 'contains(., "title")' in str(xpath)
    
    # Test 2: Contains with ident
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(element='h1'),
        MockFunction(['IDENT'], ['title'])
    )
    assert 'contains(., "title")' in str(xpath)
    
    # Test 3: Verify post_condition is added correctly
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(element='h1'),
        MockFunction(['STRING'], ['test'])
    )
    assert xpath.post_condition == 'contains(., "test")'
    
    # Test 4: Verify it raises ExpressionError for invalid argument types
    try:
        translator.xpath_contains_function(
            translator.xpathexpr_cls(element='h1'),
            MockFunction(['NUMBER'], ['1'])
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test 5: Verify it raises ExpressionError for multiple arguments
    try:
        translator.xpath_contains_function(
            translator.xpathexpr_cls(element='h1'),
            MockFunction(['STRING', 'STRING'], ['a', 'b'])
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


class MockFunction:
    """Helper class to mock the function argument in xpath_contains_function"""
    def __init__(self, argument_types, arguments):
        self._argument_types = argument_types
        self.arguments = [MockArgument(v) for v in arguments]
    
    def argument_types(self):
        return self._argument_types


class MockArgument:
    """Helper class to mock individual arguments"""
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #56
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with a different index
    xpath2 = translator.xpathexpr_cls()
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result2 = translator.xpath_eq_function(xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 3'
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    class MockFunction3:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :eq\(\)"):
        translator.xpath_eq_function(xpath3, MockFunction3())
```


# LLM-generated content at query #57
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='class="foo"'),
        type('Function', (), {
            'arguments': [type('Arg', (), {'value': '.bar'})()],
            'argument_types': lambda self: ['STRING']
        })()
    )
    result = str(xpath)
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in result
    assert 'div' in result
    
    # Test with empty result (no matching child)
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='class="foo"'),
        type('Function', (), {
            'arguments': [type('Arg', (), {'value': '.baz'})()],
            'argument_types': lambda self: ['STRING']
        })()
    )
    result = str(xpath)
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]' in result
    
    # Test with element selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='class="foo"'),
        type('Function', (), {
            'arguments': [type('Arg', (), {'value': 'div'})()],
            'argument_types': lambda self: ['IDENT']
        })()
    )
    result = str(xpath)
    assert 'descendant::div' in result
    
    # Test with invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(element='div'),
            type('Function', (), {
                'arguments': [type('Arg', (), {'value': '123'})()],
                'argument_types': lambda self: ['NUMBER']
            })()
        )


# LLM-generated content at query #58
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = lambda: None
    function.arguments = [type('arg', (), {'value': '.bar', 'type': 'STRING'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"
    
    # Test with IDENT argument type
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': 'div', 'type': 'IDENT'})()]
    function.argument_types = lambda: ['IDENT']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::div"
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': '1', 'type': 'NUMBER'})()]
    function.argument_types = lambda: ['NUMBER']
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg1', (), {'value': '.bar', 'type': 'STRING'})(),
                          type('arg2', (), {'value': '.baz', 'type': 'STRING'})()]
    function.argument_types = lambda: ['STRING', 'STRING']
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with complex selector
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': '.bar > .baz', 'type': 'STRING'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*' in result.post_condition
    assert 'bar' in result.post_condition
    assert 'baz' in result.post_condition
    
    # Test that post_condition is properly set
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': 'p', 'type': 'STRING'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with empty string selector
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': '', 'type': 'STRING'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*"


# LLM-generated content at query #59
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"test"')[0]])
    result = translator.xpath_contains_function(xpath, func)
    assert "contains(., 'test')" in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('test')[0]])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert "contains(., 'test')" in str(result2)
    
    # Test that it raises ExpressionError for invalid arguments
    import pytest
    with pytest.raises(ExpressionError):
        func3 = Function('contains', [parse('123')[0]])
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)


# LLM-generated content at query #60
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert str(result) == '*[position() = 1]'
    
    xpath2 = translator.xpathexpr_cls()
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '3'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, mock_func2)
    assert str(result2) == '*[position() = 4]'


# LLM-generated content at query #61
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction('NUMBER', '0'))
    assert str(result) == '*[position() = 1]'
    
    # Test with different index values
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_eq_function(xpath2, MockFunction('NUMBER', '2'))
    assert str(result2) == '*[position() = 3]'
    
    # Test with element and condition
    xpath3 = translator.xpathexpr_cls(element='div', condition='@class')
    result3 = translator.xpath_eq_function(xpath3, MockFunction('NUMBER', '1'))
    assert str(result3) == 'div[@class][position() = 2]'
    
    # Test error handling for non-NUMBER argument types
    from cssselect.xpath import ExpressionError
    try:
        xpath4 = translator.xpathexpr_cls()
        translator.xpath_eq_function(xpath4, MockFunction('STRING', 'test'))
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer" in str(e)
    
    # Test error handling for multiple arguments
    try:
        xpath5 = translator.xpathexpr_cls()
        translator.xpath_eq_function(xpath5, MockFunction(['NUMBER', 'NUMBER'], ['1', '2']))
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer" in str(e)


class MockFunction:
    """Helper class to mock cssselect function objects"""
    def __init__(self, argument_types, arguments):
        if isinstance(argument_types, list):
            self._argument_types = argument_types
            self.arguments = [MockArgument(arg) for arg in arguments]
        else:
            self._argument_types = [argument_types]
            self.arguments = [MockArgument(arguments)]
    
    def argument_types(self):
        return self._argument_types


class MockArgument:
    """Helper class to mock cssselect argument objects"""
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #62
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic has with class selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test 2: Has with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': 'div'})]})()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test 3: Has with IDENT argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {'argument_types': lambda: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'bar'})]})()
    )
    assert 'descendant::' in str(xpath)
    
    # Test 4: Invalid argument type raises ExpressionError
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :has"):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
        )


# LLM-generated content at query #63
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)


# LLM-generated content at query #64
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with NUMBER argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    # Create a mock function with NUMBER argument
    class MockNumber:
        def __init__(self, value):
            self.value = value
        def argument_types(self):
            return ['NUMBER']
        @property
        def arguments(self):
            return [self]
    
    function = MockNumber('1')
    xpath = XPathExpr()
    result = translator.xpath_lt_function(xpath, function)
    
    # Verify post_condition is set correctly (position() < 2 since index 1 + 1 = 2)
    assert result.post_condition == 'position() < 2'
    assert result == xpath

    # Test with a different number
    function2 = MockNumber('3')
    xpath2 = XPathExpr()
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 4'

    # Test with zero
    function3 = MockNumber('0')
    xpath3 = XPathExpr()
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 1'

    # Verify that non-NUMBER arguments raise ExpressionError
    class MockString:
        def __init__(self, value):
            self.value = value
        def argument_types(self):
            return ['STRING']
        @property
        def arguments(self):
            return [self]

    invalid_function = MockString('test')
    xpath4 = XPathExpr()
    try:
        translator.xpath_lt_function(xpath4, invalid_function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt()" in str(e)


# LLM-generated content at query #65
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('obj', (object,), {'value': 'title'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    translator2 = JQueryTranslator()
    xpath2 = translator2.xpathexpr_cls()
    
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('obj', (object,), {'value': 'content'})()]
    
    result2 = translator2.xpath_contains_function(xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'content')"
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, MockFunctionInvalid())
    
    # Test multiple post_conditions
    xpath3 = translator.xpathexpr_cls()
    xpath3.add_post_condition('position() = 1')
    result3 = translator.xpath_contains_function(xpath3, MockFunctionString())
    assert result3.post_condition == "position() = 1 and (contains(., 'title'))"


# LLM-generated content at query #66
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert 'position() > 3' in str(result)
    
    # Test with another number
    xpath2 = translator.xpathexpr_cls()
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result2 = translator.xpath_gt_function(xpath2, MockFunction2())
    assert 'position() > 1' in str(result2)
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    class MockFunction3:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, MockFunction3())
```


# LLM-generated content at query #67
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with ident argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
    )
    assert 'descendant::test' in str(xpath)
    
    # Test error for invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #68
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    function = FunctionMock(arguments=[ArgumentMock(value="title", type="STRING")])
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = FunctionMock(arguments=[ArgumentMock(value="content", type="IDENT")])
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert "contains(., 'content')" in str(result2)


# LLM-generated content at query #69
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test text'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunction())
    assert result.post_condition == "contains(., 'test text')"
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'testid'})()]
    
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'testid')"
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '42'})()]
    
    try:
        translator.xpath_contains_function(xpath3, MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident" in str(e)```


# LLM-generated content at query #70
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div', condition=''),
        type('obj', (object,), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('obj', (object,), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div', condition=''),
        type('obj', (object,), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('obj', (object,), {'value': 'span'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'span' in str(xpath)
    
    # Test with invalid arguments
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(path='//div', element='div', condition=''),
            type('obj', (object,), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('obj', (object,), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #71
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic has selector with class
    xpath1 = translator.xpath_has_function(
        XPathExpr(element='div', condition="@class='foo'"),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(@class, "bar")]' in str(xpath1)
    
    # Test 2: Has selector with tag name
    xpath2 = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath2)
    
    # Test 3: Has selector with ID
    xpath3 = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '#myid'})()]
        })()
    )
    assert 'descendant::*[@id="myid"]' in str(xpath3)
    
    # Test 4: Has selector with IDENT type
    xpath4 = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath4)
    
    # Test 5: Multiple has conditions should combine with AND
    xpath5 = translator.xpath_has_function(
        XPathExpr(element='div', condition="@class='foo'"),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'and' in str(xpath5) or 'descendant::' in str(xpath5)
    
    # Test 6: Invalid argument type should raise ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(element='div'),
            type('Function', (), {
                'argument_types': lambda: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )
    
    # Test 7: Empty has selector
    xpath7 = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': ''})()]
        })()
    )
    assert 'descendant::' in str(xpath7)


# LLM-generated content at query #72
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test basic lt functionality
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with 0 index
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with large number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='100')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 101'
    
    # Test that it raises ExpressionError for invalid arguments
    from cssselect.xpath import ExpressionError
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='invalid')]
    
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt()"):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #73
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        Function('contains', [('STRING', 'title')])
    )
    assert str(xpath) == "*[contains(., 'title')]"
    
    # Test with ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        Function('contains', [('IDENT', 'title')])
    )
    assert str(xpath) == "*[contains(., 'title')]"
    
    # Test with multiple arguments (should raise ExpressionError)
    from cssselect.xpath import ExpressionError
    try:
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            Function('contains', [('NUMBER', '1'), ('NUMBER', '2')])
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with non-string/non-ident argument (should raise ExpressionError)
    try:
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            Function('contains', [('NUMBER', '42')])
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #74
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction([MockArgument('NUMBER', '0')]))
    assert str(result) == '*[position() = 1]'
    
    # Test with valid NUMBER argument for different indices
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction([MockArgument('NUMBER', '2')]))
    assert str(result) == '*[position() = 3]'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    with pytest.raises(ExpressionError):
        xpath = translator.xpathexpr_cls()
        translator.xpath_eq_function(xpath, MockFunction([MockArgument('STRING', 'test')]))
    
    # Test that it raises ExpressionError for multiple arguments
    with pytest.raises(ExpressionError):
        xpath = translator.xpathexpr_cls()
        translator.xpath_eq_function(xpath, MockFunction([MockArgument('NUMBER', '0'), MockArgument('NUMBER', '1')]))


class MockArgument:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value


class MockFunction:
    def __init__(self, arguments):
        self.arguments = arguments
    
    def argument_types(self):
        return [arg.type for arg in self.arguments]
```


# LLM-generated content at query #75
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = Function('gt', [NumericLiteral(0)])
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 1]'
    
    # Test with different number
    xpath = translator.xpathexpr_cls()
    function = Function('gt', [NumericLiteral(2)])
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 3]'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = Function('gt', [NumericLiteral(-1)])
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 0]'
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = Function('gt', [StringLiteral('foo')])
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, function)
    
    # Test with multiple arguments (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = Function('gt', [NumericLiteral(1), NumericLiteral(2)])
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, function)
```


# LLM-generated content at query #76
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')[0].parsed[0].arguments[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == 'descendant-or-self::*[contains(., "title")]'
    
    # Test with IDENT argument
    func = Function('contains', [parse('title')[0].parsed[0].arguments[0]])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == 'descendant-or-self::*[contains(., "title")]'
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    func = Function('contains', [parse('1')[0].parsed[0].arguments[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(translator.xpathexpr_cls(), func)


# LLM-generated content at query #77
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    # Test with valid NUMBER argument
    translator = JQueryTranslator()
    xpath = XPathExpr()
    mock_function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '0'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result = translator.xpath_eq_function(xpath, mock_function)
    assert result.post_condition == 'position() = 1'
    assert result is xpath

    # Test with different index value
    xpath2 = XPathExpr()
    mock_function2 = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '5'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    result2 = translator.xpath_eq_function(xpath2, mock_function2)
    assert result2.post_condition == 'position() = 6'

    # Test with non-NUMBER argument type
    mock_function3 = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'text'})()],
        'argument_types': lambda self: ['STRING']
    })()
    
    try:
        translator.xpath_eq_function(XPathExpr(), mock_function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

    # Test with multiple arguments
    mock_function4 = type('Function', (), {
        'arguments': [
            type('Argument', (), {'value': '1'})(),
            type('Argument', (), {'value': '2'})()
        ],
        'argument_types': lambda self: ['NUMBER', 'NUMBER']
    })()
    
    try:
        translator.xpath_eq_function(XPathExpr(), mock_function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #78
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with different index
    xpath = translator.xpathexpr_cls()
    function.arguments = [MagicMock(value='3')]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 4'

    # Test with non-NUMBER argument should raise ExpressionError
    xpath = translator.xpathexpr_cls()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)
```


# LLM-generated content at query #79
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    # Test with string argument
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments[0].value = '.bar'
    
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'
    
    # Test with IDENT argument
    translator2 = JQueryTranslator()
    xpath2 = translator2.xpathexpr_cls()
    function2 = Mock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments[0].value = 'div'
    
    result2 = translator2.xpath_has_function(xpath2, function2)
    assert result2.post_condition == 'descendant::div'
    
    # Test with invalid argument type
    translator3 = JQueryTranslator()
    xpath3 = translator3.xpathexpr_cls()
    function3 = Mock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [Mock(value='invalid')]
    
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident"):
        translator3.xpath_has_function(xpath3, function3)


# LLM-generated content at query #80
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with another index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '3'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 4'
    
    # Test with non-NUMBER argument type
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    try:
        translator.xpath_eq_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [
            type('Arg', (), {'value': '1'}),
            type('Arg', (), {'value': '2'})
        ]
    })()
    
    try:
        translator.xpath_eq_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #81
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    """Test the xpath_has_function method of JQueryTranslator."""
    translator = JQueryTranslator()
    
    # Test valid string argument
    from cssselect.parser import Function
    from cssselect.parser import parse
    from cssselect.xpath import XPath
    
    # Create a mock xpath object
    xpath = XPathExpr()
    xpath.path = '/html/body/div'
    xpath.element = 'div'
    xpath.condition = ''
    
    # Create a mock function with string argument
    function = Function('has', [parse('"test"')[0]])
    
    result = translator.xpath_has_function(xpath, function)
    
    assert hasattr(result, 'post_condition')
    assert 'descendant::*' in result.post_condition
    assert 'test' in result.post_condition
    
    # Test with ident argument
    xpath2 = XPathExpr()
    xpath2.path = '/html/body/div'
    xpath2.element = 'div'
    
    function2 = Function('has', [parse('test')[0]])
    result2 = translator.xpath_has_function(xpath2, function2)
    
    assert hasattr(result2, 'post_condition')
    assert 'descendant::*' in result2.post_condition
    assert 'test' in result2.post_condition
    
    # Test with invalid argument types
    from cssselect.parser import Function as Func
    from cssselect.parser import parse as p
    
    # Test with NUMBER type (should raise error)
    function3 = Function('has', [p('123')[0]])
    try:
        translator.xpath_has_function(XPathExpr(), function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test the actual behavior with a simple selector
    xpath4 = XPathExpr()
    xpath4.path = '/html/body/div'
    xpath4.element = 'div'
    
    function4 = Function('has', [p('"bar"')[0]])
    result4 = translator.xpath_has_function(xpath4, function4)
    
    assert result4.post_condition is not None
    assert 'contains' not in result4.post_condition  # should use descendant, not contains
    
    # Test with element selector
    xpath5 = XPathExpr()
    xpath5.path = '/html/body/div'
    xpath5.element = 'div'
    
    function5 = Function('has', [p('div')[0]])
    result5 = translator.xpath_has_function(xpath5, function5)
    
    assert result5.post_condition is not None
    assert 'descendant::div' in result5.post_condition
```


# LLM-generated content at query #82
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with NUMBER argument
    from cssselect.parser import Function, parse
    func = Function('eq', [parse('0')[0]])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == '*[position() = 1]'
    
    # Test with different index
    func = Function('eq', [parse('5')[0]])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == '*[position() = 6]'
    
    # Test with negative index (should work with negative numbers)
    func = Function('eq', [parse('-1')[0]])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert str(xpath) == '*[position() = 0]'
    
    # Test with non-NUMBER argument should raise ExpressionError
    from cssselect.xpath import ExpressionError
    try:
        func = Function('eq', [parse('"string"')[0]])
        translator.xpath_eq_function(translator.xpathexpr_cls(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments should raise ExpressionError
    try:
        func = Function('eq', [parse('1')[0], parse('2')[0]])
        translator.xpath_eq_function(translator.xpathexpr_cls(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #83
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple number argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    xpath = XPathExpr()
    # Create a function mock with argument value "0"
    func = Function('gt', ['0'])
    func.arguments = [parse('0')[0].parsed_selectors[0].pseudo_class.arguments[0]]
    func.argument_types = lambda: ['NUMBER']
    
    result = translator.xpath_gt_function(xpath, func)
    assert str(result) == '*[position() > 1]'
    
    # Test with value "1"
    xpath2 = XPathExpr()
    func2 = Function('gt', ['1'])
    func2.arguments = [parse('1')[0].parsed_selectors[0].pseudo_class.arguments[0]]
    func2.argument_types = lambda: ['NUMBER']
    
    result2 = translator.xpath_gt_function(xpath2, func2)
    assert str(result2) == '*[position() > 2]'
    
    # Test with invalid argument type
    func3 = Function('gt', ['foo'])
    func3.arguments = [parse('foo')[0].parsed_selectors[0].pseudo_class.arguments[0]]
    func3.argument_types = lambda: ['IDENT']
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), func3)
```


# LLM-generated content at query #84
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with different index value
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 3'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath3, function3)
    
    # Test that it raises ExpressionError for multiple arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': ['0', '1']
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath4, function4)
```


# LLM-generated content at query #85
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic usage with string argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'span' in str(xpath)
    
    # Test raises ExpressionError for invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '5'})()]
            })()
        )


# LLM-generated content at query #86
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with NUMBER argument
    class MockFunction:
        def __init__(self, value):
            self.arguments = [type('MockArgument', (), {'value': str(value)})()]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test gt(0) - should create condition position() > 1
    result = translator.xpath_gt_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() > 1'
    
    # Test gt(1) - should create condition position() > 2
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_gt_function(xpath2, MockFunction(1))
    assert result2.post_condition == 'position() > 2'
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        
        arguments = ['invalid']
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), InvalidFunction())
```


# LLM-generated content at query #87
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
```


# LLM-generated content at query #88
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic usage with string selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test with ident selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError, match=r"Expected a single string or ident for :has\(\), got"):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #89
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with zero
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with negative number
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
```


# LLM-generated content at query #90
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    mock_xpath = XPathExpr()
    result = translator.xpath_contains_function(mock_xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'test')"
    
    # Test with IDENT argument type
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'title'})()]
    
    mock_xpath2 = XPathExpr()
    result2 = translator.xpath_contains_function(mock_xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'title')"
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '123'})()]
    
    import pytest
    with pytest.raises(ExpressionError) as exc_info:
        translator.xpath_contains_function(XPathExpr(), MockFunctionInvalid())
    assert "Expected a single string or ident for :contains()" in str(exc_info.value)


# LLM-generated content at query #91
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArg', (), {'value': 'test'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert 'contains(., "test")' in str(result)
    
    # Test with IDENT argument
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArg', (), {'value': 'test'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_contains_function(xpath, MockFunctionIdent())
    assert 'contains(., "test")' in str(result)
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArg', (), {'value': '42'})()]
    
    import pytest
    with pytest.raises(Exception):
        translator.xpath_contains_function(xpath, MockFunctionInvalid())
```


# LLM-generated content at query #92
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'title'})()]
    
    xpath = translator.xpath_expr_cls()
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument type
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('arg', (), {'value': 'content'})()]
    
    xpath = translator.xpath_expr_cls()
    result = translator.xpath_contains_function(xpath, MockFunctionIdent())
    assert result.post_condition == "contains(., 'content')"
    
    # Test error case with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '42'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, MockFunctionInvalid())
```


# LLM-generated content at query #93
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument type
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #94
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic functionality
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with IDENT type
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition is not None
    assert 'descendant::' in result2.post_condition
    
    # Test with different selectors
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    result3 = translator.xpath_has_function(xpath3, function3)
    assert result3.post_condition is not None
    assert 'descendant::' in result3.post_condition
    
    # Test invalid argument type raises error
    import pytest
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '42'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath4, function4)
    
    # Test with complex selector containing class and tag
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div.bar'})()]
    })()
    result5 = translator.xpath_has_function(xpath5, function5)
    assert result5.post_condition is not None
    assert 'descendant::' in result5.post_condition
    assert 'div' in result5.post_condition
    assert 'bar' in result5.post_condition
```


# LLM-generated content at query #95
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = Mock(arguments=[Mock(value='title', argument_types=lambda: ['STRING'])])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = Mock(arguments=[Mock(value='content', argument_types=lambda: ['IDENT'])])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'content')"
    
    # Test with invalid argument types
    xpath = translator.xpathexpr_cls()
    function = Mock(arguments=[Mock(value='123', argument_types=lambda: ['NUMBER'])])
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains"):
        translator.xpath_contains_function(xpath, function)
    
    # Test with empty string
    xpath = translator.xpathexpr_cls()
    function = Mock(arguments=[Mock(value='', argument_types=lambda: ['STRING'])])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., '')"


# LLM-generated content at query #96
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    from unittest.mock import MagicMock
    translator = JQueryTranslator()
    
    # Test case 1: Normal case with NUMBER argument
    xpath = MagicMock()
    xpath.add_post_condition = MagicMock()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '2'
    
    result = translator.xpath_lt_function(xpath, function)
    
    xpath.add_post_condition.assert_called_once_with('position() < 3')
    assert result == xpath
    
    # Test case 2: Test with value 0
    xpath = MagicMock()
    xpath.add_post_condition = MagicMock()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '0'
    
    result = translator.xpath_lt_function(xpath, function)
    
    xpath.add_post_condition.assert_called_once_with('position() < 1')
    assert result == xpath
    
    # Test case 3: Test with negative value
    xpath = MagicMock()
    xpath.add_post_condition = MagicMock()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '-1'
    
    result = translator.xpath_lt_function(xpath, function)
    
    xpath.add_post_condition.assert_called_once_with('position() < 0')
    assert result == xpath
    
    # Test case 4: Test with invalid argument type
    xpath = MagicMock()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock()]
    function.arguments[0].value = 'test'
    
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer" in str(e)


# LLM-generated content at query #97
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Create a mock XPath object
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function object with a number argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
        
        def argument_types(self):
            return ['NUMBER']
    
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with eq(0) - should add condition position() = 1
    function = MockFunction(0)
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with eq(1) - should add condition position() = 2
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction(1)
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with eq(5) - should add condition position() = 6
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction(5)
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 6'
```


# LLM-generated content at query #98
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector with string argument
    xpath = translator.xpath_has_function(
        XPathExpr(element='div', condition='@class="foo"'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(xpath)
    
    # Test has selector with ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test raise error for invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #99
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Create a mock function with NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
            
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
            
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should match elements at position > 1)
    function = MockFunction("0")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with value 2 (should match elements at position > 3)
    xpath2 = XPathExpr()
    function2 = MockFunction("2")
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with value -1 (should match elements at position > 0)
    xpath3 = XPathExpr()
    function3 = MockFunction("-1")
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), InvalidFunction())
```


# LLM-generated content at query #100
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='.bar')]
    
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = Mock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments = [Mock(value='div')]
    
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition is not None
    assert 'descendant::' in result2.post_condition
    
    # Test with invalid argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = Mock()
    function3.argument_types.return_value = ['NUMBER']
    function3.arguments = [Mock(value='1')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath3, function3)
    
    # Verify post_condition is properly added to existing condition
    xpath4 = translator.xpathexpr_cls()
    xpath4.add_post_condition('position() = 1')
    function4 = Mock()
    function4.argument_types.return_value = ['STRING']
    function4.arguments = [Mock(value='.test')]
    
    result4 = translator.xpath_has_function(xpath4, function4)
    assert 'position() = 1' in result4.post_condition
    assert 'descendant::' in result4.post_condition


# LLM-generated content at query #101
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"test"')], 'STRING')
    result = translator.xpath_contains_function(xpath, func)
    assert result.post_condition == "contains(., 'test')"
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('test')], 'IDENT')
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert result2.post_condition == "contains(., 'test')"
    
    # Test that invalid argument types raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    func3 = Function('contains', [parse('123')], 'NUMBER')
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)
```


# LLM-generated content at query #102
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple XPath and function argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with different index
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '3'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 4'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    import pytest
    from cssselect.xpath import ExpressionError
    
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath3, function3)
```


# LLM-generated content at query #103
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    def mock_function():
        pass
    
    # Test with valid number argument
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(element='h1')
    mock_function.arguments = [type('Argument', (), {'value': '0', 'type': 'NUMBER'})()]
    mock_function.argument_types = lambda: ['NUMBER']
    
    result = translator.xpath_gt_function(xpath, mock_function)
    assert 'position() > 1' in str(result)
    
    # Test with another number
    xpath2 = translator.xpathexpr_cls(element='h1')
    mock_function.arguments = [type('Argument', (), {'value': '2', 'type': 'NUMBER'})()]
    result2 = translator.xpath_gt_function(xpath2, mock_function)
    assert 'position() > 3' in str(result2)
    
    # Test with non-number argument should raise ExpressionError
    xpath3 = translator.xpathexpr_cls(element='h1')
    mock_function.arguments = [type('Argument', (), {'value': 'test', 'type': 'STRING'})()]
    mock_function.argument_types = lambda: ['STRING']
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, mock_function)
```


# LLM-generated content at query #104
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//p', element='p')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert 'position() < 3' in str(result)
    assert '//p' in str(result)


# LLM-generated content at query #105
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('gt', [parse('1')])
    xpath = translator.xpath_gt_function(XPathExpr(), func)
    assert 'position() > 2' in str(xpath)
    
    # Test with another valid number
    func = Function('gt', [parse('0')])
    xpath = translator.xpath_gt_function(XPathExpr(), func)
    assert 'position() > 1' in str(xpath)
    
    # Test with negative number
    func = Function('gt', [parse('-1')])
    xpath = translator.xpath_gt_function(XPathExpr(), func)
    assert 'position() > 0' in str(xpath)
    
    # Test with invalid argument type (non-number)
    import pytest
    from cssselect.xpath import ExpressionError
    func = Function('gt', [parse('test')])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), func)
    
    # Test with multiple arguments (should raise error)
    func = Function('gt', [parse('1'), parse('2')])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(XPathExpr(), func)
```


# LLM-generated content at query #106
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '0'})()]
    
    xpath = XPathExpr()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with different index
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '2'})()]
    
    xpath2 = XPathExpr()
    result2 = translator.xpath_eq_function(xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 3'
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    
    xpath3 = XPathExpr()
    try:
        translator.xpath_eq_function(xpath3, MockFunctionInvalid())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with negative index
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '-1'})()]
    
    xpath4 = XPathExpr()
    result4 = translator.xpath_eq_function(xpath4, MockFunctionNegative())
    assert result4.post_condition == 'position() = 0'
```


# LLM-generated content at query #107
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with different number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with invalid argument types
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #108
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test case 1: has with string argument
    xpath = XPathExpr()
    result = translator.xpath_has_function(xpath, lambda: None)
    assert result is not None
    
    # Test case 2: has with valid selector
    xpath = XPathExpr()
    mock_function = Mock()
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [Mock(value=".bar")]
    result = translator.xpath_has_function(xpath, mock_function)
    assert 'descendant::' in result.post_condition
    
    # Test case 3: has with IDENT argument
    xpath = XPathExpr()
    mock_function = Mock()
    mock_function.argument_types.return_value = ['IDENT']
    mock_function.arguments = [Mock(value="div")]
    result = translator.xpath_has_function(xpath, mock_function)
    assert 'descendant::' in result.post_condition
    
    # Test case 4: has with invalid argument type
    xpath = XPathExpr()
    mock_function = Mock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [Mock(value="1")]
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, mock_function)
    
    # Test case 5: has with multiple arguments
    xpath = XPathExpr()
    mock_function = Mock()
    mock_function.argument_types.return_value = ['STRING', 'STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, mock_function)


# LLM-generated content at query #109
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpath_lt_function(
        translator.xpathexpr_cls(element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '2'})()]
        })()
    )
    assert 'position() < 3' in str(xpath)


# LLM-generated content at query #110
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr(path='//h1', element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    
    # Test with different position
    xpath2 = XPathExpr(path='//p', element='p')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath3 = XPathExpr(path='//div', element='div')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test error case with invalid argument type
    xpath4 = XPathExpr(path='//span', element='span')
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, function4)


# LLM-generated content at query #111
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock()]
    function.arguments[0].value = '0'
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'

    function.arguments[0].value = '2'
    xpath2 = XPathExpr()
    result2 = translator.xpath_gt_function(xpath2, function)
    assert result2.post_condition == 'position() > 3'


# LLM-generated content at query #112
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with position 0 (first element)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('Argument', (), {'value': '0'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 1]'
    
    # Test with position 1 (second element)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('Argument', (), {'value': '1'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 2]'
    
    # Test with position 5
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('Argument', (), {'value': '5'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 6]'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('Argument', (), {'value': '-1'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 0]'
    
    # Test that non-NUMBER argument type raises ExpressionError
    import pytest
    from cssselect.xpath import ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['STRING']
    function.arguments = [type('Argument', (), {'value': 'test'})()]
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)
    
    # Test that multiple arguments raise ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {})()
    function.argument_types = lambda: ['NUMBER', 'NUMBER']
    function.arguments = [type('Argument', (), {'value': '0'}), type('Argument', (), {'value': '1'})]
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)
```


# LLM-generated content at query #113
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock()]
    function.arguments[0].value = '2'
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with number 0
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, function)
    function.arguments[0].value = '0'
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative number
    function.arguments[0].value = '-1'
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type (not NUMBER)
    function.argument_types.return_value = ['STRING']
    function.arguments[0].value = 'test'
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_lt_function(xpath, function)
    
    # Test with empty arguments
    function.argument_types.return_value = []
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #114
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Create a mock function with NUMBER argument
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    # Create a mock XPathExpr
    xpath = XPathExpr()
    
    # Test :eq(0) - first element (position() = 1)
    function = MockFunction("0")
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test :eq(1) - second element (position() = 2)
    xpath2 = XPathExpr()
    function2 = MockFunction("1")
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test :eq(5) - sixth element (position() = 6)
    xpath3 = XPathExpr()
    function3 = MockFunction("5")
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 6'
    
    # Test with negative index
    xpath4 = XPathExpr()
    function4 = MockFunction("-1")
    result4 = translator.xpath_eq_function(xpath4, function4)
    assert result4.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class BadFunction:
        def argument_types(self):
            return ['STRING']
        arguments = []
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), BadFunction())
```


# LLM-generated content at query #115
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse, Selector
    from cssselect.xpath import XPathExpr
    
    # Create a mock function with string argument
    function = Function('contains', [type('Argument', (), {'type': 'STRING', 'value': 'title'})()])
    
    # Create a basic xpath for testing
    xpath = translator.xpathexpr_cls(path='//h1', element='h1')
    
    result = translator.xpath_contains_function(xpath, function)
    
    # Check that post_condition contains the contains expression
    assert 'contains(.,' in result.post_condition
    assert 'title' in result.post_condition
    
    # Test with IDENT argument
    function_ident = Function('contains', [type('Argument', (), {'type': 'IDENT', 'value': 'text'})()])
    
    xpath2 = translator.xpathexpr_cls(path='//p', element='p')
    result2 = translator.xpath_contains_function(xpath2, function_ident)
    
    assert 'contains(.,' in result2.post_condition
    assert 'text' in result2.post_condition
    
    # Test with invalid argument types
    function_invalid = Function('contains', [type('Argument', (), {'type': 'NUMBER', 'value': '123'})()])
    
    xpath3 = translator.xpathexpr_cls(path='//div', element='div')
    try:
        translator.xpath_contains_function(xpath3, function_invalid)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #116
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result.post_condition == 'position() < 3'
    
    # Test with negative number
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '-1'})()]
    
    result2 = translator.xpath_lt_function(xpath2, MockFunctionNegative())
    assert result2.post_condition == 'position() < 0'
    
    # Test with zero
    xpath3 = translator.xpathexpr_cls()
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result3 = translator.xpath_lt_function(xpath3, MockFunctionZero())
    assert result3.post_condition == 'position() < 1'
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    try:
        translator.xpath_lt_function(translator.xpathexpr_cls(), MockFunctionInvalid())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    class MockFunctionMultiple:
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
        arguments = [type('MockArgument', (), {'value': '1'})()]
    
    try:
        translator.xpath_lt_function(translator.xpathexpr_cls(), MockFunctionMultiple())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with non-integer number
    class MockFunctionFloat:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2.5'})()]
    
    try:
        translator.xpath_lt_function(translator.xpathexpr_cls(), MockFunctionFloat())
        assert False, "Expected ValueError"
    except ValueError:
        pass
```


# LLM-generated content at query #117
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func_str = Function('contains', [parse('"title"')[0].parsed_tree])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func_str)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with IDENT argument
    func_ident = Function('contains', [parse('title')[0].parsed_tree])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func_ident)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with invalid argument type
    from cssselect.parser import Function
    func_invalid = Function('contains', [parse('1')[0].parsed_tree])
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls(), func_invalid)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert 'Expected a single string or ident for :contains()' in str(e)


# LLM-generated content at query #118
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    from cssselect.parser import parse, Function
    from cssselect.xpath import ExpressionError
    
    translator = JQueryTranslator()
    
    # Test with simple xpath and valid number argument
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = Function('eq', [('NUMBER', '0')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//div[position() = 1]'
    
    # Test with different index
    xpath = translator.xpathexpr_cls(path='//h1', element='h1')
    function = Function('eq', [('NUMBER', '3')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//h1[position() = 4]'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls(path='//p', element='p')
    function = Function('eq', [('NUMBER', '-1')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//p[position() = 0]'
    
    # Test with zero index
    xpath = translator.xpathexpr_cls(path='//span', element='span')
    function = Function('eq', [('NUMBER', '0')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//span[position() = 1]'
    
    # Test that non-number argument raises ExpressionError
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = Function('eq', [('STRING', 'invalid')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)
    
    # Test that multiple arguments raise ExpressionError
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = Function('eq', [('NUMBER', '1'), ('NUMBER', '2')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)
    
    # Test with large number
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = Function('eq', [('NUMBER', '999')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '//div[position() = 1000]'
```


# LLM-generated content at query #119
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test basic lt with number argument
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '1'))
    assert str(result) == '*[position() < 2]'
    
    # Test lt with positional index 0
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '0'))
    assert str(result) == '*[position() < 1]'
    
    # Test lt with negative index
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '-1'))
    assert str(result) == '*[position() < 0]'
    
    # Test lt with large number
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '100'))
    assert str(result) == '*[position() < 101]'
    
    # Test that invalid argument type raises ExpressionError
    import pytest
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, MockFunction(['STRING'], 'test'))
    
    # Test that multiple arguments raise ExpressionError
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, MockFunction(['NUMBER', 'NUMBER'], '1, 2'))


# LLM-generated content at query #120
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Create a mock function with NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should select elements with position > 1)
    xpath = XPathExpr()
    function = MockFunction("0")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with value 2 (should select elements with position > 3)
    xpath = XPathExpr()
    function = MockFunction("2")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with negative value (-1 should select elements with position > 0)
    xpath = XPathExpr()
    function = MockFunction("-1")
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockFunctionNonNumber:
        def __init__(self):
            self.arguments = ["string_arg"]
        
        def argument_types(self):
            return ['STRING']
    
    xpath = XPathExpr()
    function = MockFunctionNonNumber()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #121
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with different number
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test that it raises ExpressionError for invalid argument types
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath3, function3)


# LLM-generated content at query #122
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test case 1: select first element (index 0)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 1]'
    
    # Test case 2: select second element (index 1)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 2]'
    
    # Test case 3: select element with negative index (should work as xpath position)
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*[position() = 0]'
    
    # Test case 4: check that non-NUMBER argument types raise ExpressionError
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #123
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
    assert "contains(., 'title')" in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'content'})()]
        })()
    )
    assert "contains(., 'content')" in str(xpath)
    
    # Test with invalid argument type raises ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #124
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    # Setup
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    xpath.element = 'h1'
    
    # Create a mock function object
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, args):
            self.arguments = args
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should match positions > 1)
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with value 1 (should match positions > 2)
    function = MockFunction([MockArgument('1')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'
    
    # Test with value 5 (should match positions > 6)
    function = MockFunction([MockArgument('5')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 6'
    
    # Test with negative value
    function = MockFunction([MockArgument('-1')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'  # -1 + 1 = 0
    
    # Test with value 0 on a new xpath
    xpath2 = translator.xpathexpr_cls()
    xpath2.element = 'div'
    function = MockFunction([MockArgument('0')])
    result = translator.xpath_gt_function(xpath2, function)
    assert result.post_condition == 'position() > 1'  # 0 + 1 = 1
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockFunctionInvalid:
        def __init__(self, args):
            self.arguments = args
        
        def argument_types(self):
            return ['STRING']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, MockFunctionInvalid([MockArgument('test')]))
```


# LLM-generated content at query #125
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    mock_xpath = XPathExpr()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument type
    mock_xpath2 = XPathExpr()
    mock_function2 = type('MockFunction', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('MockArgument', (), {'value': 'content'})()]
    })()
    
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function2)
    assert "contains(., 'content')" in str(result2)
    
    # Test that it raises ExpressionError for invalid argument types
    from cssselect.xpath import ExpressionError
    import pytest
    
    mock_xpath3 = XPathExpr()
    mock_function3 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '5'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath3, mock_function3)


# LLM-generated content at query #126
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with positional index 0 (should match positions > 1)
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Arg', (), {'value': '0'})()]
        })()
    )
    assert 'position() > 1' in str(xpath)
    
    # Test with positional index 2 (should match positions > 3)
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Arg', (), {'value': '2'})()]
        })()
    )
    assert 'position() > 3' in str(xpath)
    
    # Test with negative index
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['NUMBER'],
            'arguments': [type('Arg', (), {'value': '-1'})()]
        })()
    )
    assert 'position() > 0' in str(xpath)
```


# LLM-generated content at query #127
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    """Test xpath_contains_function with various inputs."""
    translator = JQueryTranslator()
    
    # Test with STRING argument
    mock_function_string = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'title'})()]
    })()
    
    mock_xpath = type('XPathExpr', (), {
        'post_condition': None,
        'add_post_condition': lambda self, cond: setattr(self, 'post_condition', cond)
    })()
    
    result = translator.xpath_contains_function(mock_xpath, mock_function_string)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    mock_function_ident = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'content'})()]
    })()
    
    mock_xpath2 = type('XPathExpr', (), {
        'post_condition': None,
        'add_post_condition': lambda self, cond: setattr(self, 'post_condition', cond)
    })()
    
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function_ident)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test with invalid argument type
    mock_function_invalid = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '42'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath, mock_function_invalid)


# LLM-generated content at query #128
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'test')"
    
    # Test with invalid argument type
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '42'})()]
    })()
    try:
        translator.xpath_contains_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [
            type('Argument', (), {'value': 'foo'})(),
            type('Argument', (), {'value': 'bar'})()
        ]
    })()
    try:
        translator.xpath_contains_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #129
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with a simple case
    from cssselect.parser import Function, parse
    function = parse(':gt(2)')[0]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with 0
    xpath2 = translator.xpathexpr_cls()
    function2 = parse(':gt(0)')[0]
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 1'
    
    # Test with negative value
    xpath3 = translator.xpathexpr_cls()
    function3 = parse(':gt(-1)')[0]
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
```


# LLM-generated content at query #130
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
    assert "contains(., 'title')" in str(xpath)
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'text'})()]
        })()
    )
    assert "contains(., 'text')" in str(xpath)
    
    # Test with invalid argument type
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
    
    # Test with multiple arguments
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['STRING', 'STRING'],
                'arguments': [
                    type('Argument', (), {'value': 'a'}),
                    type('Argument', (), {'value': 'b'})
                ]
            })()
        )


# LLM-generated content at query #131
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, Function("has", [String("div")]))
    assert "descendant::div" in str(result)
    
    # Test has with class selector
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, Function("has", [String(".bar")]))
    assert "descendant::*[contains" in str(result) or "descendant::*[@class" in str(result)
    
    # Test has with multiple conditions
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_has_function(xpath, Function("has", [String("div.baz")]))
    assert "descendant::div" in str(result)
    
    # Test that post_condition is added
    xpath = translator.xpathexpr_cls()
    translator.xpath_has_function(xpath, Function("has", [String("p")]))
    assert xpath.post_condition is not None
    
    # Test invalid argument type raises ExpressionError
    import pytest
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, Function("has", [Number("1")]))


# LLM-generated content at query #132
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test with large value
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '100'})()]
    })()
    
    result4 = translator.xpath_lt_function(xpath4, function4)
    assert result4.post_condition == 'position() < 101'
    
    # Test invalid argument types raise ExpressionError
    import pytest
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath5, function5)


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_disabled_pseudo():
    translator = JQueryTranslator()
    
    # Test with a simple input element
    xpath = translator.xpath_disabled_pseudo(translator.xpathexpr_cls())
    assert "@disabled" in str(xpath)
    
    # Test that the xpath contains the disabled condition
    xpath_str = str(xpath)
    assert "disabled" in xpath_str
    assert "input" in xpath_str
    assert "button" in xpath_str
    assert "fieldset" in xpath_str
    assert "optgroup" in xpath_str
    assert "option" in xpath_str
    
    # Test that the condition is added correctly
    assert xpath.condition is not None
    assert "disabled" in xpath.condition
    
    # Test with an existing condition
    xpath2 = translator.xpathexpr_cls(condition="@type = 'text'")
    xpath2 = translator.xpath_disabled_pseudo(xpath2)
    assert "@type = 'text'" in str(xpath2)
    assert "disabled" in str(xpath2)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'title'})()]
        })()
    )
    assert str(xpath) == "*[contains(., 'title')]"
    
    # Test with IDENT argument type
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
    )
    assert str(xpath) == "*[contains(., 'test')]"
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains()"):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})]
        })()
    )
    assert hasattr(xpath, 'post_condition')
    assert 'descendant::' in xpath.post_condition
    assert 'bar' in xpath.post_condition

    # Test with ident argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})]
        })()
    )
    assert hasattr(xpath, 'post_condition')
    assert 'descendant::' in xpath.post_condition
    assert 'div' in xpath.post_condition

    # Test raises ExpressionError for invalid argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})]
            })()
        )

    # Test with compound selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div.bar'})]
        })()
    )
    assert hasattr(xpath, 'post_condition')
    assert 'descendant::' in xpath.post_condition
    assert 'div' in xpath.post_condition
    assert 'bar' in xpath.post_condition

    # Test post_condition is applied correctly
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='/test', element='div'),
        type('Function', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.foo'})]
        })()
    )
    assert xpath.post_condition is not None
    assert 'descendant::' in str(xpath) or 'descendant::' in xpath.post_condition
```


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test text'})()]
    
    mock_xpath = XPathExpr()
    result = translator.xpath_contains_function(mock_xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'test text')"
    
    # Test with ident argument
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'test_id'})()]
    
    mock_xpath2 = XPathExpr()
    result2 = translator.xpath_contains_function(mock_xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'test_id')"
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '5'})()]
    
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains"):
        translator.xpath_contains_function(XPathExpr(), MockFunctionInvalid())


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    
    # Test with input element
    xpath_input = translator.xpathexpr_cls(element='input')
    result_input = translator.xpath_input_pseudo(xpath_input)
    assert "name(.) = 'input'" in str(result_input)
    assert "name(.) = 'select'" in str(result_input)
    assert "name(.) = 'textarea'" in str(result_input)
    assert "name(.) = 'button'" in str(result_input)
    
    # Test with select element
    xpath_select = translator.xpathexpr_cls(element='select')
    result_select = translator.xpath_input_pseudo(xpath_select)
    assert "name(.) = 'input'" in str(result_select)
    assert "name(.) = 'select'" in str(result_select)
    assert "name(.) = 'textarea'" in str(result_select)
    assert "name(.) = 'button'" in str(result_select)
    
    # Test with textarea element
    xpath_textarea = translator.xpathexpr_cls(element='textarea')
    result_textarea = translator.xpath_input_pseudo(xpath_textarea)
    assert "name(.) = 'input'" in str(result_textarea)
    assert "name(.) = 'select'" in str(result_textarea)
    assert "name(.) = 'textarea'" in str(result_textarea)
    assert "name(.) = 'button'" in str(result_textarea)
    
    # Test with button element
    xpath_button = translator.xpathexpr_cls(element='button')
    result_button = translator.xpath_input_pseudo(xpath_button)
    assert "name(.) = 'input'" in str(result_button)
    assert "name(.) = 'select'" in str(result_button)
    assert "name(.) = 'textarea'" in str(result_button)
    assert "name(.) = 'button'" in str(result_button)
    
    # Test that the method returns the same xpath object
    xpath = translator.xpathexpr_cls(element='input')
    result = translator.xpath_input_pseudo(xpath)
    assert result is xpath
```


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = XPathExpr('//h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with argument value 1 (second element)
    xpath2 = XPathExpr('//h1')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    xpath3 = XPathExpr('//h1')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_eq_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test with invalid argument type (not NUMBER)
    xpath = translator.xpathexpr_cls()
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
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'}),
                      type('Argument', (), {'value': '2'})]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpath_input_pseudo(
        translator.xpathexpr_cls(element='*')
    )
    expected_condition = (
        "(name(.) = 'input' or name(.) = 'select') "
        "or (name(.) = 'textarea' or name(.) = 'button')"
    )
    assert xpath.condition == expected_condition
```


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    
    # Test with input element
    xpath_input = translator.xpathexpr_cls(element='input')
    result_input = translator.xpath_input_pseudo(xpath_input)
    assert "name(.) = 'input'" in str(result_input)
    assert "name(.) = 'select'" in str(result_input)
    assert "name(.) = 'textarea'" in str(result_input)
    assert "name(.) = 'button'" in str(result_input)
    
    # Test with select element
    xpath_select = translator.xpathexpr_cls(element='select')
    result_select = translator.xpath_input_pseudo(xpath_select)
    assert "name(.) = 'input'" in str(result_select)
    assert "name(.) = 'select'" in str(result_select)
    assert "name(.) = 'textarea'" in str(result_select)
    assert "name(.) = 'button'" in str(result_select)
    
    # Test with textarea element
    xpath_textarea = translator.xpathexpr_cls(element='textarea')
    result_textarea = translator.xpath_input_pseudo(xpath_textarea)
    assert "name(.) = 'input'" in str(result_textarea)
    assert "name(.) = 'select'" in str(result_textarea)
    assert "name(.) = 'textarea'" in str(result_textarea)
    assert "name(.) = 'button'" in str(result_textarea)
    
    # Test with button element
    xpath_button = translator.xpathexpr_cls(element='button')
    result_button = translator.xpath_input_pseudo(xpath_button)
    assert "name(.) = 'input'" in str(result_button)
    assert "name(.) = 'select'" in str(result_button)
    assert "name(.) = 'textarea'" in str(result_button)
    assert "name(.) = 'button'" in str(result_button)
    
    # Test that the returned object is the same instance
    assert result_input is xpath_input
    assert result_select is xpath_select
    assert result_textarea is xpath_textarea
    assert result_button is xpath_button
    
    # Test that condition is properly formatted
    condition = str(result_input).split('[')[-1].rstrip(']')
    assert condition == "((name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button'))"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, mock_function)
    assert result.post_condition == 'position() < 3'
    
    # Test with different number value
    xpath2 = translator.xpathexpr_cls()
    mock_function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, mock_function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test that it raises ExpressionError with non-number argument
    xpath3 = translator.xpathexpr_cls()
    mock_function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath3, mock_function3)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another number
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test with non-number argument should raise ExpressionError
    import pytest
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), function4)


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr(path='//p', element='p')
    xpath = translator.xpath_gt_function(xpath, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })())
    assert str(xpath) == "//p[position() > 1]"
    
    # Test with non-zero index
    xpath2 = XPathExpr(path='//div', element='div')
    xpath2 = translator.xpath_gt_function(xpath2, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })())
    assert str(xpath2) == "//div[position() > 3]"
    
    # Test with negative index
    xpath3 = XPathExpr(path='//span', element='span')
    xpath3 = translator.xpath_gt_function(xpath3, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })())
    assert str(xpath3) == "//span[position() > 0]"
    
    # Test with existing post_condition
    xpath4 = XPathExpr(path='//li', element='li')
    xpath4.post_condition = 'position() < 5'
    xpath4 = translator.xpath_gt_function(xpath4, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })())
    assert str(xpath4) == "//li[position() < 5][position() > 2]"
    
    # Test error case with non-NUMBER argument
    import pytest
    with pytest.raises(ExpressionError):
        xpath5 = XPathExpr(path='//a', element='a')
        translator.xpath_gt_function(xpath5, type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })())
```


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid input
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['2'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with 0
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['0'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['-1'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['STRING'], ['test'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER', 'NUMBER'], ['1', '2'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with non-integer number
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['3.5'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

class MockFunction:
    def __init__(self, argument_types, arguments):
        self._argument_types = argument_types
        self.arguments = arguments
    
    def argument_types(self):
        return self._argument_types
```


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()],
        'value': '.bar'
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", @class, " "), " bar ")]' in str(result)
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()],
        'value': 'div'
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert 'descendant::div' in str(result2)
    
    # Test error case with wrong argument types
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '42'})()],
        'value': '42'
    })()
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments (should raise error)
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Arg', (), {'value': 'foo'}),
                     type('Arg', (), {'value': 'bar'})],
        'value': 'foo, bar'
    })()
    try:
        translator.xpath_has_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    # Mock a function with STRING argument
    class MockFunction:
        def __init__(self, value, arg_type):
            self.arguments = [type('MockArgument', (), {'value': value})()]
            self._arg_types = [arg_type]
        
        def argument_types(self):
            return self._arg_types
    
    # Test with string argument
    func = MockFunction("title", 'STRING')
    xpath = XPathExpr()
    result = translator.xpath_contains_function(xpath, func)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    func = MockFunction("title", 'IDENT')
    xpath = XPathExpr()
    result = translator.xpath_contains_function(xpath, func)
    assert "contains(., 'title')" in str(result)
    
    # Test that it raises ExpressionError for invalid argument types
    import pytest
    func = MockFunction("1", 'NUMBER')
    xpath = XPathExpr()
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :contains()"):
        translator.xpath_contains_function(xpath, func)


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with IDENT argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument types
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Arg', (), {'value': 'foo'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"
    
    # Test with element selector
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition == "descendant::div"
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '5'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :has"):
        translator.xpath_has_function(xpath3, function3)


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    mock_xpath = XPathExpr(path='//div', element='div', condition='', star_prefix=False)
    mock_function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'title', 'type': 'STRING'})()],
        'argument_types': lambda self: ['STRING']
    })()
    
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert "contains(., 'title')" in result.post_condition
    
    # Test with ident argument
    mock_xpath2 = XPathExpr(path='//h1', element='h1', condition='', star_prefix=False)
    mock_function2 = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'content', 'type': 'IDENT'})()],
        'argument_types': lambda self: ['IDENT']
    })()
    
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function2)
    assert "contains(., 'content')" in result2.post_condition
    
    # Test with invalid argument type
    mock_xpath3 = XPathExpr(path='//*', element='*', condition='', star_prefix=False)
    mock_function3 = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 123, 'type': 'NUMBER'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    
    try:
        translator.xpath_contains_function(mock_xpath3, mock_function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    mock_xpath4 = XPathExpr(path='//*', element='*', condition='', star_prefix=False)
    mock_function4 = type('Function', (), {
        'arguments': [
            type('Argument', (), {'value': 'text', 'type': 'STRING'})(),
            type('Argument', (), {'value': 'extra', 'type': 'STRING'})()
        ],
        'argument_types': lambda self: ['STRING', 'STRING']
    })()
    
    try:
        translator.xpath_contains_function(mock_xpath4, mock_function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr('div', 'div', '', False)
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    assert str(result) == 'div[position() = 1]'
    
    # Test with different index
    xpath2 = XPathExpr('div', 'div', '', False)
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '3'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 4'
    assert str(result2) == 'div[position() = 4]'
    
    # Test with negative value (still works)
    xpath3 = XPathExpr('div', 'div', '', False)
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 0'
    assert str(result3) == 'div[position() = 0]'
```


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    
    # Create a mock function with STRING argument
    mock_function = type('MockFunction', (), {})()
    mock_function.arguments = [type('MockArgument', (), {'value': 'title', 'type': 'STRING'})()]
    mock_function.argument_types = lambda: ['STRING']
    
    # Create a mock XPath expression
    xpath = XPathExpr()
    
    result = translator.xpath_contains_function(xpath, mock_function)
    
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    mock_function2 = type('MockFunction', (), {})()
    mock_function2.arguments = [type('MockArgument', (), {'value': 'text', 'type': 'IDENT'})()]
    mock_function2.argument_types = lambda: ['IDENT']
    
    xpath2 = XPathExpr()
    result2 = translator.xpath_contains_function(xpath2, mock_function2)
    
    assert result2.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    mock_function3 = type('MockFunction', (), {})()
    mock_function3.arguments = [type('MockArgument', (), {'value': '42', 'type': 'NUMBER'})()]
    mock_function3.argument_types = lambda: ['NUMBER']
    
    xpath3 = XPathExpr()
    
    try:
        translator.xpath_contains_function(xpath3, mock_function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    mock_function4 = type('MockFunction', (), {})()
    mock_function4.arguments = [type('MockArgument', (), {'value': 'hello', 'type': 'STRING'}),
                                type('MockArgument', (), {'value': 'world', 'type': 'STRING'})]
    mock_function4.argument_types = lambda: ['STRING', 'STRING']
    
    xpath4 = XPathExpr()
    
    try:
        translator.xpath_contains_function(xpath4, mock_function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it returns the same xpath object
    assert result is xpath
    assert result2 is xpath2
    
    # Test with empty string
    mock_function5 = type('MockFunction', (), {})()
    mock_function5.arguments = [type('MockArgument', (), {'value': '', 'type': 'STRING'})()]
    mock_function5.argument_types = lambda: ['STRING']
    
    xpath5 = XPathExpr()
    result5 = translator.xpath_contains_function(xpath5, mock_function5)
    
    assert result5.post_condition == "contains(., '')"


# LLM-generated content at query #21
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
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    assert 'bar' in result.post_condition
    
    # Test with ident argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert result2.post_condition is not None
    assert 'descendant::' in result2.post_condition
    
    # Test with invalid argument types
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 0})()]
    })()
    try:
        translator.xpath_has_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is properly added
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.test'})()]
    })()
    result4 = translator.xpath_has_function(xpath4, function4)
    assert result4.post_condition == translator.css_to_xpath('.test', prefix='descendant::')```


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 1'
    
    # Test with another number
    xpath = translator.xpathexpr_cls()
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '3'})()]
    result = translator.xpath_gt_function(xpath, MockFunction2())
    assert result.post_condition == 'position() > 4'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = []
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, MockFunctionInvalid())
```


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    mock_xpath = XPathExpr()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArg', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with ident argument
    mock_xpath2 = XPathExpr()
    mock_function2 = type('MockFunction', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('MockArg', (), {'value': 'test'})()]
    })()
    
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function2)
    assert result2.post_condition == "contains(., 'test')"
    
    # Test with invalid argument type
    mock_xpath3 = XPathExpr()
    mock_function3 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '42'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath3, mock_function3)
    
    # Test with mixed argument types (should fail)
    mock_xpath4 = XPathExpr()
    mock_function4 = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING', 'NUMBER'],
        'arguments': [type('MockArg', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath4, mock_function4)


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 1'
    
    # Test with invalid argument type
    class MockInvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = []
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, MockInvalidFunction())


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    mock_xpath = type('MockXPath', (), {'post_condition': None})()
    mock_xpath.add_post_condition = lambda cond: setattr(mock_xpath, 'post_condition', cond)
    
    result = translator.xpath_eq_function(mock_xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with value 1 (should become position() = 2)
    mock_xpath2 = type('MockXPath', (), {'post_condition': None})()
    mock_xpath2.add_post_condition = lambda cond: setattr(mock_xpath2, 'post_condition', cond)
    
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '1'})()]
    
    result2 = translator.xpath_eq_function(mock_xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 2'
    
    # Test with non-NUMBER argument type
    class MockFunction3:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(mock_xpath, MockFunction3())
```


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Create a mock XPathExpr with add_post_condition method
    class MockXPath:
        def __init__(self):
            self.post_conditions = []
        def add_post_condition(self, condition):
            self.post_conditions.append(condition)
    
    xpath = MockXPath()
    
    # Test with number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '2'})()]
    
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert xpath.post_conditions == ['position() < 3']
    assert result is xpath
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = ['invalid']
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError, match="Expected a single integer"):
        translator.xpath_lt_function(MockXPath(), InvalidFunction())
```


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    # Setup
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['2'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'  # 2 + 1 = 3
    
    # Test with index 0
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['0'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with negative index
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER'], ['-1'])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['STRING'], ['test'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = MockFunction(['NUMBER', 'NUMBER'], ['1', '2'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


class MockFunction:
    """Helper class to simulate cssselect function objects"""
    def __init__(self, argument_types, arguments):
        self.argument_types = lambda: argument_types
        self.arguments = [MockArgument(arg) for arg in arguments]


class MockArgument:
    """Helper class to simulate cssselect argument objects"""
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::' in result.post_condition
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert 'descendant::' in result2.post_condition
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath3, function3)


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a mock XPath object
    class MockXPath:
        def __init__(self):
            self.post_condition = None
            self.path = ''
            self.element = '*'
            self.condition = ''
        
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    xpath = MockXPath()
    
    # Create a mock function with NUMBER argument
    class MockFunction:
        def __init__(self, value):
            self.arguments = [type('MockArgument', (), {'value': str(value)})()]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should convert to position() = 1)
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() = 1'
    
    # Test with value 1 (should convert to position() = 2)
    xpath = MockXPath()
    result = translator.xpath_eq_function(xpath, MockFunction(1))
    assert result.post_condition == 'position() = 2'
    
    # Test with value 5 (should convert to position() = 6)
    xpath = MockXPath()
    result = translator.xpath_eq_function(xpath, MockFunction(5))
    assert result.post_condition == 'position() = 6'
    
    # Test that it returns the xpath object
    xpath = MockXPath()
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result is xpath
```


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with a simple string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"
    
    # Test with an ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test that it raises ExpressionError for invalid argument types
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with index 1
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='1')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'
    
    # Test with index 5
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='5')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 6'
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)
    
    # Test with multiple arguments (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER', 'NUMBER']
    function.arguments = [MagicMock(value='0'), MagicMock(value='1')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)
```


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')[0].parsed])
    result = translator.xpath_contains_function(xpath, func)
    assert result.post_condition == "contains(., 'title')" or result.post_condition == 'contains(., "title")'
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('title')[0].parsed])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert result2.post_condition is not None
    assert 'contains' in result2.post_condition
    
    # Test that it raises ExpressionError for invalid arguments
    xpath3 = translator.xpathexpr_cls()
    try:
        from cssselect.parser import Function
        # Try to create a function with invalid arguments
        func3 = Function('contains', [parse('1')[0].parsed])
        translator.xpath_contains_function(xpath3, func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is properly set
    xpath4 = translator.xpathexpr_cls()
    func4 = Function('contains', [parse('"test"')[0].parsed])
    result4 = translator.xpath_contains_function(xpath4, func4)
    assert result4.post_condition is not None
    assert 'contains' in result4.post_condition
    
    # Test multiple calls to add_post_condition
    xpath5 = translator.xpathexpr_cls()
    func5 = Function('contains', [parse('"first"')[0].parsed])
    result5 = translator.xpath_contains_function(xpath5, func5)
    assert result5.post_condition is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('eq', [parse('0')])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert 'position() = 1' in str(xpath)
    
    # Test with another number
    func = Function('eq', [parse('5')])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert 'position() = 6' in str(xpath)
    
    # Test with negative number
    func = Function('eq', [parse('-1')])
    xpath = translator.xpath_eq_function(translator.xpathexpr_cls(), func)
    assert 'position() = 0' in str(xpath)
    
    # Test that it raises ExpressionError for non-number arguments
    func = Function('eq', [parse('"string"')])
    try:
        translator.xpath_eq_function(translator.xpathexpr_cls(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for ident arguments
    func = Function('eq', [parse('ident')])
    try:
        translator.xpath_eq_function(translator.xpathexpr_cls(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for multiple arguments
    func = Function('eq', [parse('1'), parse('2')])
    try:
        translator.xpath_eq_function(translator.xpathexpr_cls(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    xpath = XPathExpr()
    func = Function('contains', [parse('"title"')[0].parsed_selectors[0].pseudo_class.arguments[0]])
    # Actually creating a proper Function object is complex, let's use a simpler approach
    
    # Simple test using the internal structure
    class MockArgument:
        def __init__(self, value, type='STRING'):
            self.value = value
            self.type = type
        
        def argument_types(self):
            return [self.type]
    
    class MockFunction:
        def __init__(self, args):
            self.arguments = args
        
        def argument_types(self):
            return [arg.type for arg in self.arguments]
    
    # Test with STRING type
    xpath1 = XPathExpr()
    func1 = MockFunction([MockArgument('title', 'STRING')])
    result = translator.xpath_contains_function(xpath1, func1)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT type
    xpath2 = XPathExpr()
    func2 = MockFunction([MockArgument('text', 'IDENT')])
    result = translator.xpath_contains_function(xpath2, func2)
    assert "contains(., 'text')" in str(result)
    
    # Test with invalid argument type
    xpath3 = XPathExpr()
    func3 = MockFunction([MockArgument('42', 'NUMBER')])
    try:
        translator.xpath_contains_function(xpath3, func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = XPathExpr()
    func4 = MockFunction([MockArgument('hello', 'STRING'), MockArgument('world', 'STRING')])
    try:
        translator.xpath_contains_function(xpath4, func4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #35
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    # Test with STRING argument
    xpath = XPathExpr(path='//h1', element='h1')
    function = Function('contains', [parse('"title"')[0]])
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == "//h1[contains(., 'title')]"
    
    # Test with IDENT argument
    xpath = XPathExpr(path='//h1', element='h1')
    function = Function('contains', [parse('title')[0]])
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == "//h1[contains(., 'title')]"
    
    # Test with invalid argument type
    xpath = XPathExpr(path='//h1', element='h1')
    function = Function('contains', [parse('1')[0]])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)
    
    # Test with multiple arguments
    xpath = XPathExpr(path='//h1', element='h1')
    function = Function('contains', [parse('"a"')[0], parse('"b"')[0]])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with valid STRING argument
    xpath = translator.xpathexpr_cls()
    function = lambda: None
    function.arguments = [type('arg', (), {'value': '.bar'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with valid IDENT argument
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': 'div'})()]
    function.argument_types = lambda: ['IDENT']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with invalid argument types
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': 'test'})()]
    function.argument_types = lambda: ['NUMBER']
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with no post_condition initially
    xpath = translator.xpathexpr_cls()
    function.arguments = [type('arg', (), {'value': '.test'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    assert '.test' in result.post_condition
    
    # Test with existing post_condition
    xpath = translator.xpathexpr_cls()
    xpath.post_condition = "existing_condition"
    function.arguments = [type('arg', (), {'value': '.test'})()]
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert 'existing_condition' in result.post_condition
    assert 'and' in result.post_condition
    assert 'descendant::' in result.post_condition
```


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = XPathExpr(path='//div', element='div')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with different values
    xpath2 = XPathExpr(path='//p', element='p')
    function2 = MagicMock()
    function2.argument_types.return_value = ['NUMBER']
    function2.arguments = [MagicMock(value='0')]
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test error handling with invalid argument type
    function3 = MagicMock()
    function3.argument_types.return_value = ['STRING']
    function3.arguments = [MagicMock(value='invalid')]
    
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function3)


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import XPathExpr
    
    xpath = XPathExpr()
    function = Function('eq', [parse('0')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "descendant-or-self::*[position() = 1]"
    
    xpath = XPathExpr()
    function = Function('eq', [parse('5')])
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "descendant-or-self::*[position() = 6]"
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    xpath = XPathExpr()
    function = Function('eq', [parse('"string"')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    xpath = XPathExpr()
    function = Function('eq', [parse('identifier')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments (should fail)
    xpath = XPathExpr()
    function = Function('eq', [parse('1'), parse('2')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with different number
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '3'})()]
    
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_eq_function(xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 4'
    
    # Test with non-number argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single integer for :eq()"):
        translator.xpath_eq_function(translator.xpathexpr_cls(), MockFunctionInvalid())


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with a simple number argument
    from cssselect.parser import Function, parse
    func = Function('gt', [parse('1')[0]])
    result = translator.xpath_gt_function(xpath, func)
    assert result.post_condition == 'position() > 2'
    
    # Test with zero
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('gt', [parse('0')[0]])
    result2 = translator.xpath_gt_function(xpath2, func2)
    assert result2.post_condition == 'position() > 1'
    
    # Test with negative number (though not typical, should still work)
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('gt', [parse('-1')[0]])
    result3 = translator.xpath_gt_function(xpath3, func3)
    assert result3.post_condition == 'position() > 0'
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    with pytest.raises(ExpressionError):
        func4 = Function('gt', [parse('"string"')[0]])
        translator.xpath_gt_function(translator.xpathexpr_cls(), func4)
```


# LLM-generated content at query #41
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock Function object with NUMBER argument type
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with index 0
    function = MockFunction('0')
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'
    
    # Test with index 1
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction('1')
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 2'
    
    # Test with index 5
    xpath3 = translator.xpathexpr_cls()
    function3 = MockFunction('5')
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 6'
    
    # Test with negative index (should still work, though not standard)
    xpath4 = translator.xpathexpr_cls()
    function4 = MockFunction('-1')
    result4 = translator.xpath_eq_function(xpath4, function4)
    assert result4.post_condition == 'position() = 0'
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockStringFunction:
        def __init__(self):
            self.arguments = ['test']
        
        def argument_types(self):
            return ['STRING']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(translator.xpathexpr_cls(), MockStringFunction())
```


# LLM-generated content at query #42
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple XPath and function with number argument
    xpath = XPathExpr(path='//div/p', element='p')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    
    assert result.post_condition == 'position() > 1'
    assert str(result) == '//div/p[position() > 1]'
    
    # Test with value 2
    xpath2 = XPathExpr(path='//div/p', element='p')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    
    assert result2.post_condition == 'position() > 3'
    assert str(result2) == '//div/p[position() > 3]'
```


# LLM-generated content at query #43
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with valid string argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'title'})()]
        })()
    )
    assert "contains(., 'title')" in str(xpath)
    
    # Test with valid ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'text'})()]
        })()
    )
    assert "contains(., 'text')" in str(xpath)
    
    # Test with invalid argument type
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '42'})()]
            })()
        )


# LLM-generated content at query #44
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple case
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, type('FakeFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('FakeArgument', (), {'value': '2'})]
    })())
    assert str(result) == '*[position() < 3]'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, type('FakeFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('FakeArgument', (), {'value': '0'})]
    })())
    assert str(result) == '*[position() < 1]'
    
    # Test with large number
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, type('FakeFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('FakeArgument', (), {'value': '100'})]
    })())
    assert str(result) == '*[position() < 101]'
    
    # Test with negative number (though unlikely in practice)
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, type('FakeFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('FakeArgument', (), {'value': '-1'})]
    })())
    assert str(result) == '*[position() < 0]'
    
    # Test with element and condition preserved
    xpath = translator.xpathexpr_cls(element='div', condition='@class')
    result = translator.xpath_lt_function(xpath, type('FakeFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('FakeArgument', (), {'value': '5'})]
    })())
    assert str(result) == 'div[@class][position() < 6]'
```


# LLM-generated content at query #45
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a mock xpath object
    class MockXPath:
        def __init__(self):
            self.post_condition = None
            
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    xpath = MockXPath()
    
    # Mock function with NUMBER argument type
    class MockFunction:
        def __init__(self, value):
            self.arguments = [type('MockArgument', (), {'value': value})()]
            
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should give position() < 1)
    function = MockFunction('0')
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with value 1 (should give position() < 2)
    xpath2 = MockXPath()
    function2 = MockFunction('1')
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 2'
    
    # Test with value 5 (should give position() < 6)
    xpath3 = MockXPath()
    function3 = MockFunction('5')
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 6'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    class MockInvalidFunction:
        def __init__(self):
            self.arguments = [type('MockArgument', (), {'value': 'test'})()]
            
        def argument_types(self):
            return ['STRING']
    
    invalid_function = MockInvalidFunction()
    try:
        translator.xpath_lt_function(MockXPath(), invalid_function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #46
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a valid number argument
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction(["NUMBER"], ["2"]))
    assert str(result) == '*[position() = 3]'
    
    # Test with first element (index 0)
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction(["NUMBER"], ["0"]))
    assert str(result) == '*[position() = 1]'
    
    # Test with invalid argument type
    import pytest
    from cssselect.xpath import ExpressionError
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, MockFunction(["STRING"], ["test"]))
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, MockFunction(["NUMBER", "NUMBER"], ["1", "2"]))


# LLM-generated content at query #47
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    class MockFunctionFirst:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunctionFirst())
    assert result.post_condition == 'position() = 1'
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'text'})()]
    
    import pytest
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, MockFunctionInvalid())
```


# LLM-generated content at query #48
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument type
    from cssselect.parser import Function, parse
    func1 = Function('contains', [parse('"title"')[0].parsed])
    result = translator.xpath_contains_function(xpath, func1)
    assert 'contains(., "title")' in str(result)
    
    # Test with IDENT argument type  
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('hello')[0].parsed])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert 'contains(., "hello")' in str(result2)
    
    # Test raises ExpressionError for invalid argument types
    import pytest
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('contains', [parse('123')[0].parsed])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, func3)


# LLM-generated content at query #49
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('lt', [parse('2')[0]])
    result = translator.xpath_lt_function(xpath, func)
    assert result.post_condition == 'position() < 3'
    
    # Test with 0 index
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('lt', [parse('0')[0]])
    result2 = translator.xpath_lt_function(xpath2, func2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('lt', [parse('-1')[0]])
    result3 = translator.xpath_lt_function(xpath3, func3)
    assert result3.post_condition == 'position() < 0'
    
    # Test with non-number argument should raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    xpath4 = translator.xpathexpr_cls()
    func4 = Function('lt', [parse('"abc"')[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, func4)
```


# LLM-generated content at query #50
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')[0].parsed])
    xpath = translator.xpath_contains_function(XPathExpr(), func)
    assert "contains(., 'title')" in str(xpath)
    
    # Test with IDENT argument
    func = Function('contains', [parse('title')[0].parsed])
    xpath = translator.xpath_contains_function(XPathExpr(), func)
    assert "contains(., 'title')" in str(xpath)
    
    # Test with invalid argument types
    from cssselect.parser import Function
    func = Function('contains', [])
    try:
        translator.xpath_contains_function(XPathExpr(), func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test actual behavior with PyQuery
    from pyquery import PyQuery
    d = PyQuery('<div><h1/><h1 class="title">title</h1></div>')
    result = d('h1:contains("title")')
    assert len(result) == 1
    assert result[0].get('class') == 'title'
```


# LLM-generated content at query #51
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with xpath that has no existing conditions
    result = translator.xpath_gt_function(xpath, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})]
    })())
    assert str(result).endswith('[position() > 1]')
    
    # Test with different index values
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_gt_function(xpath2, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})]
    })())
    assert str(result2).endswith('[position() > 3]')
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'text'})]
        })())
    
    # Test with existing post_condition
    xpath3 = translator.xpathexpr_cls()
    xpath3.add_post_condition('position() = 1')
    result3 = translator.xpath_gt_function(xpath3, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})]
    })())
    assert 'position() = 1' in str(result3)
    assert 'position() > 1' in str(result3)
```


# LLM-generated content at query #52
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        def __init__(self, value):
            self.arguments = [type('Argument', (), {'value': str(value)})()]
    
    # Test eq(0) - first element
    xpath = XPathExpr()
    result = translator.xpath_eq_function(xpath, MockFunction(0))
    assert result.post_condition == 'position() = 1'
    
    # Test eq(1) - second element
    xpath = XPathExpr()
    result = translator.xpath_eq_function(xpath, MockFunction(1))
    assert result.post_condition == 'position() = 2'
    
    # Test eq(5) - sixth element
    xpath = XPathExpr()
    result = translator.xpath_eq_function(xpath, MockFunction(5))
    assert result.post_condition == 'position() = 6'
    
    # Test that it raises ExpressionError for non-NUMBER argument
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        def __init__(self):
            self.arguments = [type('Argument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr(), MockFunctionInvalid())
```


# LLM-generated content at query #53
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        class Argument:
            def __init__(self, value):
                self.value = value
        arguments = [Argument('0')]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 1'
    
    # Test with different number
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        class Argument:
            def __init__(self, value):
                self.value = value
        arguments = [Argument('5')]
    
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_eq_function(xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 6'
    
    # Test with negative number
    class MockFunction3:
        def argument_types(self):
            return ['NUMBER']
        class Argument:
            def __init__(self, value):
                self.value = value
        arguments = [Argument('-1')]
    
    xpath3 = translator.xpathexpr_cls()
    result3 = translator.xpath_eq_function(xpath3, MockFunction3())
    assert result3.post_condition == 'position() = 0'
```


# LLM-generated content at query #54
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='title')]
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath2 = translator.xpathexpr_cls()
    function2 = MagicMock()
    function2.argument_types.return_value = ['IDENT']
    function2.arguments = [MagicMock(value='content')]
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test invalid argument type raises ExpressionError
    xpath3 = translator.xpathexpr_cls()
    function3 = MagicMock()
    function3.argument_types.return_value = ['NUMBER']
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, function3)
    
    # Test with empty string
    xpath4 = translator.xpathexpr_cls()
    function4 = MagicMock()
    function4.argument_types.return_value = ['STRING']
    function4.arguments = [MagicMock(value='')]
    
    result4 = translator.xpath_contains_function(xpath4, function4)
    assert result4.post_condition == "contains(., '')"


# LLM-generated content at query #55
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with a valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with another valid number
    xpath2 = XPathExpr()
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '0'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, mock_func2)
    assert result2.post_condition == 'position() = 1'
    
    # Test with non-number argument types
    xpath3 = XPathExpr()
    mock_func3 = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_eq_function(xpath3, mock_func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with non-integer number argument
    xpath4 = XPathExpr()
    mock_func4 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '1.5'})()]
    })()
    try:
        translator.xpath_eq_function(xpath4, mock_func4)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```


# LLM-generated content at query #56
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'title'})()]
        })()
    )
    assert "contains(., 'title')" in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'content'})()]
        })()
    )
    assert "contains(., 'content')" in str(xpath)
    
    # Test raises ExpressionError for invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '42'})()]
            })()
        )
```


# LLM-generated content at query #57
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function with a NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        class Argument:
            def __init__(self, value):
                self.value = value
        arguments = [Argument('2')]
    
    function = MockFunction()
    result = translator.xpath_lt_function(xpath, function)
    
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    function2 = MockFunction()
    function2.arguments[0].value = '0'
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'


# LLM-generated content at query #58
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with a simple xpath
    result = translator.xpath_gt_function(xpath, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })())
    
    assert result.post_condition == 'position() > 1'
    assert result is xpath  # Should return the same xpath object
    
    # Test with value 2
    xpath2 = XPathExpr()
    result2 = translator.xpath_gt_function(xpath2, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })())
    
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative value
    xpath3 = XPathExpr()
    result3 = translator.xpath_gt_function(xpath3, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })())
    
    assert result3.post_condition == 'position() > 0'
    
    # Test with different xpath element
    xpath4 = XPathExpr(element='div')
    result4 = translator.xpath_gt_function(xpath4, type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })())
    
    assert result4.post_condition == 'position() > 6'
    assert result4.element == 'div'
```


# LLM-generated content at query #59
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'
    
    # Test that ExpressionError is raised for invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    function_invalid = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(translator.xpathexpr_cls(), function_invalid)


# LLM-generated content at query #60
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArg', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 3'
    
    # Test with invalid argument types
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = ['invalid']
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, InvalidFunction())


# LLM-generated content at query #61
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), 
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'title'})()]
        })()
    )
    assert str(xpath) == "descendant-or-self::*[contains(., 'title')]"
    
    # Test with IDENT argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'content'})()]
        })()
    )
    assert str(xpath) == "descendant-or-self::*[contains(., 'content')]"
```


# LLM-generated content at query #62
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    from cssselect.parser import Function, Token
    string_func = Function(
        'has',
        [Token('STRING', '"test"', 0, 0)]
    )
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        string_func
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " test ")]' in str(xpath)
    
    # Test with IDENT argument type
    ident_func = Function(
        'has',
        [Token('IDENT', 'div', 0, 0)]
    )
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        ident_func
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with complex selector
    complex_func = Function(
        'has',
        [Token('STRING', '"div.foo"', 0, 0)]
    )
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        complex_func
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " foo ")]' in str(xpath)
    
    # Test error case - wrong argument type
    from cssselect.xpath import ExpressionError
    import pytest
    wrong_func = Function(
        'has',
        [Token('NUMBER', '123', 0, 0)]
    )
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            wrong_func
        )


# LLM-generated content at query #63
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid positive number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with large number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='5')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 6'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #64
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'bar'})()]
        })()
    )
    assert 'descendant::bar' in str(xpath)
    assert 'contains' not in str(xpath)
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
    )
    assert 'descendant::test' in str(xpath)
    
    # Test that post_condition is added correctly
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'foo'})()]
        })()
    )
    assert str(xpath).endswith('[descendant::foo]')
    
    # Test error case
    try:
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #65
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and number argument
    xpath = XPathExpr()
    xpath.element = 'h1'
    
    # Create a mock function with NUMBER argument
    class MockArg:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        
        def __init__(self, args):
            self.arguments = args
    
    function = MockFunction([MockArg('1')])
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'
    assert result.element == 'h1'
    
    # Test with value 0
    xpath2 = XPathExpr()
    xpath2.element = 'div'
    function2 = MockFunction([MockArg('0')])
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with value 5
    xpath3 = XPathExpr()
    xpath3.element = 'p'
    function3 = MockFunction([MockArg('5')])
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 6'
```


# LLM-generated content at query #66
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument type
    from cssselect.parser import Function, Token
    string_func = Function('contains', [Token('STRING', '"title"')])
    result = translator.xpath_contains_function(xpath, string_func)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    ident_func = Function('contains', [Token('IDENT', 'title')])
    result2 = translator.xpath_contains_function(xpath2, ident_func)
    assert "contains(., 'title')" in str(result2)
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    xpath3 = translator.xpathexpr_cls()
    invalid_func = Function('contains', [Token('NUMBER', '123')])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, invalid_func)


# LLM-generated content at query #67
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    # Test with index 2 (should map to position() = 3)
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert str(result) == '*[position() = 3]'
    assert result.post_condition == 'position() = 3'
    
    # Test with index 0 (should map to position() = 1)
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunctionZero())
    assert str(result) == '*[position() = 1]'
    assert result.post_condition == 'position() = 1'
    
    # Test with negative index (should still work)
    class MockFunctionNeg:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '-1'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_eq_function(xpath, MockFunctionNeg())
    assert str(result) == '*[position() = 0]'
    assert result.post_condition == 'position() = 0'
    
    # Test that invalid argument type raises ExpressionError
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    xpath = translator.xpathexpr_cls()
    try:
        translator.xpath_eq_function(xpath, InvalidFunction())
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #68
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath object 
    xpath = XPathExpr()
    xpath.element = 'h1'
    
    # Create a mock function object that simulates a NUMBER argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        
        def __init__(self):
            self.arguments = []
            
            class MockArgument:
                def __init__(self, value):
                    self.value = value
            
            self.arguments.append(MockArgument('2'))
    
    function = MockFunction()
    result = translator.xpath_lt_function(xpath, function)
    
    # Check that post_condition is correctly set for position() < 3 (value + 1)
    assert result.post_condition == 'position() < 3'
    assert isinstance(result, XPathExpr)

    # Test with value 0 (should give position() < 1)
    xpath2 = XPathExpr()
    xpath2.element = 'h1'
    
    function2 = MockFunction()
    function2.arguments[0].value = '0'
    result2 = translator.xpath_lt_function(xpath2, function2)
    
    assert result2.post_condition == 'position() < 1'

    # Test that it raises ExpressionError for non-NUMBER argument types
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        
        def __init__(self):
            self.arguments = []
            
            class MockArgument:
                def __init__(self, value):
                    self.value = value
            
            self.arguments.append(MockArgument('test'))
    
    function3 = MockFunctionString()
    try:
        translator.xpath_lt_function(XPathExpr(), function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass  # Expected
```


# LLM-generated content at query #69
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and number argument
    xpath = translator.xpathexpr_cls()
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction)
    assert str(result) == '*[position() > 1]'
    
    # Test with different index
    xpath = translator.xpathexpr_cls()
    mock_func = MockFunction()
    mock_func.arguments[0].value = '2'
    result = translator.xpath_gt_function(xpath, mock_func)
    assert str(result) == '*[position() > 3]'
    
    # Test error handling with wrong argument type
    xpath = translator.xpathexpr_cls()
    class BadFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, BadFunction())
```


# LLM-generated content at query #70
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)
```


# LLM-generated content at query #71
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector with string argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'class' in str(xpath) or 'bar' in str(xpath)
    
    # Test has with ident argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with empty selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': ''})()]
        })()
    )
    assert 'descendant::*' in str(xpath) or 'descendant::' in str(xpath)
    
    # Test has with invalid argument type
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


# LLM-generated content at query #72
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Argument', (), {'value': '2'})()]
    
    class MockXPath:
        def __init__(self):
            self.post_condition = None
        
        def add_post_condition(self, condition):
            self.post_condition = condition
    
    xpath = MockXPath()
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    class MockFunctionFirst:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Argument', (), {'value': '0'})()]
    
    xpath2 = MockXPath()
    result2 = translator.xpath_eq_function(xpath2, MockFunctionFirst())
    assert result2.post_condition == 'position() = 1'
    
    # Test with negative number
    class MockFunctionNegative:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Argument', (), {'value': '-1'})()]
    
    xpath3 = MockXPath()
    result3 = translator.xpath_eq_function(xpath3, MockFunctionNegative())
    assert result3.post_condition == 'position() = 0'
    
    # Test with non-number argument should raise ExpressionError
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Argument', (), {'value': 'foo'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(MockXPath(), MockFunctionInvalid())
```


# LLM-generated content at query #73
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath and function argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 1'
    
    # Test with negative index
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_eq_function(xpath3, function3)
    assert result3.post_condition == 'position() = 0'
    
    # Test that non-NUMBER argument types raise ExpressionError
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath4, function4)
    
    # Test that multiple arguments raise ExpressionError
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'}), type('Arg', (), {'value': '2'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath5, function5)


# LLM-generated content at query #74
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with class selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.foo'})()]
        })()
    )
    assert 'descendant::*' in str(xpath)
    assert 'foo' in str(xpath)
    
    # Test has with ID selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '#myid'})()]
        })()
    )
    assert 'descendant::*' in str(xpath)
    assert 'myid' in str(xpath)


# LLM-generated content at query #75
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test basic lt function
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(), 
                                        type('function', (), {
                                            'argument_types': lambda self: ['NUMBER'],
                                            'arguments': [type('arg', (), {'value': '1'})()]
                                        })())
    assert 'position() < 2' in str(xpath)
    
    # Test lt with index 0
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(),
                                        type('function', (), {
                                            'argument_types': lambda self: ['NUMBER'],
                                            'arguments': [type('arg', (), {'value': '0'})()]
                                        })())
    assert 'position() < 1' in str(xpath)
    
    # Test lt with large index
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(),
                                        type('function', (), {
                                            'argument_types': lambda self: ['NUMBER'],
                                            'arguments': [type('arg', (), {'value': '5'})()]
                                        })())
    assert 'position() < 6' in str(xpath)
    
    # Test lt with negative index should raise error
    import pytest
    with pytest.raises(ValueError):
        translator.xpath_lt_function(translator.xpathexpr_cls(),
                                    type('function', (), {
                                        'argument_types': lambda self: ['NUMBER'],
                                        'arguments': [type('arg', (), {'value': '-1'})()]
                                    })())
    
    # Test lt with non-number argument should raise ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(translator.xpathexpr_cls(),
                                    type('function', (), {
                                        'argument_types': lambda self: ['STRING'],
                                        'arguments': [type('arg', (), {'value': 'test'})()]
                                    })())
```


# LLM-generated content at query #76
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = XPathExpr(path='//h1', element='h1')
    from unittest.mock import Mock
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='0')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another number
    xpath2 = XPathExpr(path='//h1', element='h1')
    function2 = Mock()
    function2.argument_types.return_value = ['NUMBER']
    function2.arguments = [Mock(value='2')]
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with invalid argument type raises ExpressionError
    xpath3 = XPathExpr(path='//h1', element='h1')
    function3 = Mock()
    function3.argument_types.return_value = ['STRING']
    function3.arguments = [Mock(value='invalid')]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, function3)


# LLM-generated content at query #77
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Create a mock function object with argument_types and arguments
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Arg', (), {'value': '2'})()]
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    
    assert result.post_condition == 'position() = 3'  # 2 + 1 = 3
    assert result == xpath  # Should return the same xpath object

    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    class MockFunction2:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Arg', (), {'value': '0'})()]
    
    result2 = translator.xpath_eq_function(xpath2, MockFunction2())
    assert result2.post_condition == 'position() = 1'  # 0 + 1 = 1

    # Test with non-NUMBER argument type raises error
    class MockFunction3:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Arg', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(translator.xpathexpr_cls(), MockFunction3())
```


# LLM-generated content at query #78
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse
    from cssselect.xpath import ExpressionError
    
    # Test successful case with STRING
    selector = ':contains("test text")'
    func = Function('contains', [parse('"test text"').parsed_selectors[0].pseudo_class.arguments[0]])
    result = translator.xpath_contains_function(xpath, func)
    assert "contains(., 'test text')" in str(result)
    
    # Test successful case with IDENT
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('contains', [parse('test').parsed_selectors[0].pseudo_class.arguments[0]])
    result2 = translator.xpath_contains_function(xpath2, func2)
    assert "contains(., 'test')" in str(result2)
    
    # Test with invalid argument type (should raise ExpressionError)
    from cssselect.parser import parse as css_parse
    try:
        func3 = Function('contains', [parse(':first').parsed_selectors[0].pseudo_class])
        translator.xpath_contains_function(translator.xpathexpr_cls(), func3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with NUMBER argument (should raise ExpressionError)
    try:
        func4 = Function('contains', [parse('123').parsed_selectors[0].pseudo_class.arguments[0]])
        translator.xpath_contains_function(translator.xpathexpr_cls(), func4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #79
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')[0]])
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with ident argument (no quotes)
    func = parse(':contains(foo)')[0]
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), func)
    assert 'contains(., "foo")' in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    from cssselect.xpath import ExpressionError
    
    # Test with multiple arguments
    func = Function('contains', [parse('"a"')[0], parse('"b"')[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(), func)
    
    # Test with number argument
    func = Function('contains', [parse('123')[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(), func)
```


# LLM-generated content at query #80
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    from cssselect.parser import Function, parse
    func = Function('lt', [parse('3')[0].parsed_tree])
    result = translator.xpath_lt_function(xpath, func)
    assert result.post_condition == 'position() < 4'  # lt(3) means index < 3, position < 4
    
    # Test with negative number
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('lt', [parse('-1')[0].parsed_tree])
    result2 = translator.xpath_lt_function(xpath2, func2)
    assert result2.post_condition == 'position() < 0'
    
    # Test with zero
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('lt', [parse('0')[0].parsed_tree])
    result3 = translator.xpath_lt_function(xpath3, func3)
    assert result3.post_condition == 'position() < 1'  # lt(0) means index < 0, position < 1
    
    # Test that it raises ExpressionError for non-number arguments
    from cssselect.xpath import ExpressionError
    try:
        func4 = Function('lt', [parse('"string"')[0].parsed_tree])
        translator.xpath_lt_function(translator.xpathexpr_cls(), func4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it returns the xpath object
    xpath5 = translator.xpathexpr_cls()
    func5 = Function('lt', [parse('5')[0].parsed_tree])
    returned = translator.xpath_lt_function(xpath5, func5)
    assert returned is xpath5  # Should return the same xpath object
```


# LLM-generated content at query #81
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test basic lt(1) - should match elements with position < 2 (index 0)
    xpath = translator.xpath_lt_function(
        cssselect_xpath.XPathExpr(path='//h1', element='h1'),
        MockFunction(arguments=[MockArgument(value='1', type='NUMBER')])
    )
    assert str(xpath) == "//h1[position() < 2]"
    
    # Test lt(0) - should match nothing (position < 1)
    xpath = translator.xpath_lt_function(
        cssselect_xpath.XPathExpr(path='//h1', element='h1'),
        MockFunction(arguments=[MockArgument(value='0', type='NUMBER')])
    )
    assert str(xpath) == "//h1[position() < 1]"
    
    # Test lt(5) - should match elements with position < 6
    xpath = translator.xpath_lt_function(
        cssselect_xpath.XPathExpr(path='//div', element='div'),
        MockFunction(arguments=[MockArgument(value='5', type='NUMBER')])
    )
    assert str(xpath) == "//div[position() < 6]"
    
    # Test with non-NUMBER argument type should raise ExpressionError
    try:
        translator.xpath_lt_function(
            cssselect_xpath.XPathExpr(path='//p', element='p'),
            MockFunction(arguments=[MockArgument(value='test', type='STRING')])
        )
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments should raise ExpressionError
    try:
        translator.xpath_lt_function(
            cssselect_xpath.XPathExpr(path='//p', element='p'),
            MockFunction(arguments=[
                MockArgument(value='1', type='NUMBER'),
                MockArgument(value='2', type='NUMBER')
            ])
        )
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


class MockFunction:
    def __init__(self, arguments=None):
        self.arguments = arguments or []
    
    def argument_types(self):
        return [arg.type for arg in self.arguments]


class MockArgument:
    def __init__(self, value='', type=''):
        self.value = value
        self.type = type
```


# LLM-generated content at query #82
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test with ident type argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #83
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, MockFunction())
    
    assert result.post_condition == 'position() > 3'  # position() > (2+1)
    assert isinstance(result, XPathExpr)
    
    # Test with invalid argument type
    class InvalidFunction:
        def argument_types(self):
            return ['STRING']
        arguments = []
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, InvalidFunction())
```


# LLM-generated content at query #84
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function(monkeypatch):
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='/div/h1')
    
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    
    assert str(result) == '/div/h1[position() > 1]'
    
    # Test with value 2
    xpath2 = translator.xpathexpr_cls(path='/div/h1')
    mock_func2 = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArgument', (), {'value': '2'})()]
    })()
    result2 = translator.xpath_gt_function(xpath2, mock_func2)
    assert str(result2) == '/div/h1[position() > 3]'
    
    # Test with non-NUMBER argument type
    xpath3 = translator.xpathexpr_cls(path='/div/h1')
    mock_func3 = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (), {'value': '0'})()]
    })()
    try:
        translator.xpath_gt_function(xpath3, mock_func3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #85
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, Number
    xpath = translator.xpathexpr_cls()
    function = Function('lt', [Number('2')])
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 3]'
    
    # Test with 0 index
    xpath = translator.xpathexpr_cls()
    function = Function('lt', [Number('0')])
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 1]'
    
    # Test raises ExpressionError for invalid argument types
    from cssselect.parser import Function, Ident
    xpath = translator.xpathexpr_cls()
    function = Function('lt', [Ident('abc')])
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)
```


# LLM-generated content at query #86
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Create a mock XPath expression
    xpath = XPathExpr()
    
    # Create a mock function with STRING argument type
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('obj', (object,), {'value': 'test text'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunction())
    assert result.post_condition == "contains(., 'test text')"
    
    # Test with IDENT argument type
    xpath2 = XPathExpr()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('obj', (object,), {'value': 'test'})()]
    
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'test')"
    
    # Test with invalid argument type
    import pytest
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '123'})()]
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, MockFunctionInvalid())
    
    # Test that original xpath is modified and returned
    xpath3 = XPathExpr()
    result3 = translator.xpath_contains_function(xpath3, MockFunction())
    assert result3 is xpath3
```


# LLM-generated content at query #87
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    """Test the xpath_has_function method of JQueryTranslator."""
    
    # Create a translator instance
    translator = JQueryTranslator()
    
    # Create a mock XPath expression
    xpath = XPathExpr()
    
    # Test with STRING argument type
    function_mock = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    
    result = translator.xpath_has_function(xpath, function_mock)
    assert result.post_condition is not None
    assert 'descendant::' in result.post_condition
    
    # Test with IDENT argument type
    xpath2 = XPathExpr()
    function_mock2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    
    result2 = translator.xpath_has_function(xpath2, function_mock2)
    assert result2.post_condition is not None
    assert 'descendant::' in result2.post_condition
    
    # Test with invalid argument type should raise ExpressionError
    xpath3 = XPathExpr()
    function_mock3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    
    try:
        translator.xpath_has_function(xpath3, function_mock3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments should raise ExpressionError
    xpath4 = XPathExpr()
    function_mock4 = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Arg', (), {'value': 'test'}), type('Arg', (), {'value': 'test2'})]
    })()
    
    try:
        translator.xpath_has_function(xpath4, function_mock4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #88
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic usage with string selector
    from cssselect.parser import parse, Function
    parsed = parse(':has(".bar")')
    selector = parsed[0]
    pseudo_class = selector.pseudo_element
    
    # Create a mock xpath object
    mock_xpath = XPathExpr()
    
    # Test with valid string argument
    result = translator.xpath_has_function(mock_xpath, pseudo_class)
    assert result is mock_xpath
    assert mock_xpath.post_condition is not None
    assert 'descendant::' in mock_xpath.post_condition
    assert 'bar' in mock_xpath.post_condition
    
    # Test that post_condition is properly added
    mock_xpath2 = XPathExpr()
    result2 = translator.xpath_has_function(mock_xpath2, pseudo_class)
    assert result2 is mock_xpath2
    assert 'descendant::' in mock_xpath2.post_condition
    
    # Test with invalid argument type
    from cssselect.parser import Function as Func
    invalid_func = Func('has', [])
    try:
        translator.xpath_has_function(mock_xpath, invalid_func)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with IDENT argument
    mock_xpath3 = XPathExpr()
    ident_func = Func('has', [type('Arg', (), {'value': 'div', 'type': 'IDENT'})()])
    translator.xpath_has_function(mock_xpath3, ident_func)
    assert 'descendant::' in mock_xpath3.post_condition
    assert 'div' in mock_xpath3.post_condition
    
    # Test that post_condition is properly formatted
    mock_xpath4 = XPathExpr()
    translator.xpath_has_function(mock_xpath4, pseudo_class)
    assert mock_xpath4.post_condition.startswith('descendant::')
```


# LLM-generated content at query #89
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

def test_JQueryTranslator_xpath_lt_function_negative():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

def test_JQueryTranslator_xpath_lt_function_invalid_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #90
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    from cssselect.parser import Function, parse
    func = Function('has', [parse('".bar"')[0].parsed_selectors[0]])
    xpath = translator.xpath_has_function(translator.xpathexpr_cls(), func)
    assert 'descendant::' in str(xpath)
    assert '.bar' in str(xpath) or 'bar' in str(xpath)
    
    # Test with IDENT argument type
    func2 = Function('has', [parse('div')[0].parsed_selectors[0]])
    xpath2 = translator.xpath_has_function(translator.xpathexpr_cls(), func2)
    assert 'descendant::' in str(xpath2)
    assert 'div' in str(xpath2)
    
    # Test that invalid argument types raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    func3 = Function('has', [parse(':first')[0].parsed_selectors[0]])
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(translator.xpathexpr_cls(), func3)
```


# LLM-generated content at query #91
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test text'})()]
    
    xpath = XPathExpr()
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'test text')"
    
    # Test with IDENT argument
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'test_text'})()]
    
    xpath = XPathExpr()
    result = translator.xpath_contains_function(xpath, MockFunctionIdent())
    assert result.post_condition == "contains(., 'test_text')"
    
    # Test with invalid argument type
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '42'})()]
    
    xpath = XPathExpr()
    try:
        translator.xpath_contains_function(xpath, MockFunctionInvalid())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    class MockFunctionMultiple:
        def argument_types(self):
            return ['STRING', 'STRING']
        arguments = [type('MockArgument', (), {'value': 'text1'}), 
                     type('MockArgument', (), {'value': 'text2'})()]
    
    xpath = XPathExpr()
    try:
        translator.xpath_contains_function(xpath, MockFunctionMultiple())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with empty string
    class MockFunctionEmpty:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': ''})()]
    
    xpath = XPathExpr()
    result = translator.xpath_contains_function(xpath, MockFunctionEmpty())
    assert result.post_condition == "contains(., '')"```


# LLM-generated content at query #92
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid NUMBER argument
    from cssselect.parser import Function, Number
    func = Function('lt', [Number('2')])
    result = translator.xpath_lt_function(xpath, func)
    assert result.post_condition == 'position() < 3'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    func2 = Function('lt', [Number('0')])
    result2 = translator.xpath_lt_function(xpath2, func2)
    assert result2.post_condition == 'position() < 1'
    
    # Test with invalid argument type
    from cssselect.parser import String
    xpath3 = translator.xpathexpr_cls()
    func3 = Function('lt', [String('string_arg')])
    try:
        translator.xpath_lt_function(xpath3, func3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls()
    func4 = Function('lt', [Number('1'), Number('2')])
    try:
        translator.xpath_lt_function(xpath4, func4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #93
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(2))
    assert str(result) == '*[position() < 3]'
    
    # Test with element specified
    xpath = translator.xpathexpr_cls(element='h1')
    result = translator.xpath_lt_function(xpath, MockFunction(1))
    assert str(result) == 'h1[position() < 2]'
    
    # Test with existing condition
    xpath = translator.xpathexpr_cls(element='div', condition='@class')
    result = translator.xpath_lt_function(xpath, MockFunction(0))
    assert str(result) == 'div[@class][position() < 1]'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(-1))
    assert str(result) == '*[position() < 0]'


class MockFunction:
    def __init__(self, value):
        self.arguments = [MockArgument(value)]
    
    def argument_types(self):
        return ['NUMBER']


class MockArgument:
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #94
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock()]
    function.arguments[0].value = '2'
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, function)
    
    assert result.post_condition == 'position() > 3'
    assert result == xpath

    # Test with negative value
    function.arguments[0].value = '-1'
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'

    # Test with invalid argument type
    function.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #95
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    from cssselect.parser import Function, parse as css_parse
    parsed = css_parse(':contains("title")')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., 'title')" in result
    
    # Test with IDENT argument (no quotes)
    parsed = css_parse(':contains(title)')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., 'title')" in result
    
    # Test with single character string
    parsed = css_parse(':contains("a")')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., 'a')" in result
    
    # Test with numeric string
    parsed = css_parse(':contains("123")')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., '123')" in result
    
    # Test with empty string
    parsed = css_parse(':contains("")')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., '')" in result
    
    # Test with special characters
    parsed = css_parse(':contains("hello world")')
    selector = parsed[0]
    function = selector.pseudo_element
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(), function
    )
    result = str(xpath)
    assert "contains(., 'hello world')" in result
    
    # Test that invalid argument types raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    
    # Mock function with invalid argument type
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = []
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(), MockFunction()
        )```


# LLM-generated content at query #96
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'title'})()]
    
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_contains_function(xpath, MockFunction())
    assert 'contains(., "title")' in str(result)
    
    # Test with IDENT argument
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArgument', (), {'value': 'text'})()]
    
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert 'contains(., "text")' in str(result2)
    
    # Test that it raises ExpressionError for invalid argument types
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '5'})()]
    
    import pytest
    xpath3 = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, MockFunctionInvalid())
    
    # Test with empty arguments
    class MockFunctionEmpty:
        def argument_types(self):
            return []
        arguments = []
    
    xpath4 = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath4, MockFunctionEmpty())


# LLM-generated content at query #97
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='title')]
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'title')" in str(result)
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['IDENT']
    function.arguments = [Mock(value='test')]
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., 'test')" in str(result)
    
    # Test with invalid argument types
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value=5)]
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)
    
    # Test with empty argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='')]
    result = translator.xpath_contains_function(xpath, function)
    assert "contains(., '')" in str(result)


# LLM-generated content at query #98
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    # Test with STRING argument type
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    class MockFunctionString:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArg', (), {'value': 'title'})()]
    result = translator.xpath_contains_function(xpath, MockFunctionString())
    assert result.post_condition == "contains(., 'title')"

    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('MockArg', (), {'value': 'text'})()]
    result2 = translator.xpath_contains_function(xpath2, MockFunctionIdent())
    assert result2.post_condition == "contains(., 'text')"

    # Test with invalid argument type (should raise ExpressionError)
    xpath3 = translator.xpathexpr_cls()
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArg', (), {'value': '42'})()]
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath3, MockFunctionInvalid())
```


# LLM-generated content at query #99
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test has with tag selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'div' in str(xpath)
    
    # Test invalid argument type raises error
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '42'})()]
            })()
        )
```


# LLM-generated content at query #100
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with different number value
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
    try:
        translator.xpath_gt_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #101
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test case 1: Basic lt(0) - should match nothing (first element is position 1)
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '0'))
    assert str(result) == '*[position() < 1]'
    
    # Test case 2: lt(1) - should match first element
    xpath = translator.xpathexpr_cls()
    xpath.add_post_condition('position() < 2')
    
    # Test case 3: lt(2) - should match first and second elements
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, MockFunction(['NUMBER'], '2'))
    assert str(result) == '*[position() < 3]'
    
    # Test case 4: Verify error is raised for non-numeric arguments
    try:
        translator.xpath_lt_function(translator.xpathexpr_cls(), MockFunction(['STRING'], 'foo'))
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

class MockFunction:
    def __init__(self, types, value):
        self.arguments = [MockArgument(value)]
        self._types = types
    
    def argument_types(self):
        return self._types

class MockArgument:
    def __init__(self, value):
        self.value = value
```


# LLM-generated content at query #102
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    from cssselect.parser import Function
    from cssselect.parser import parse
    
    # Mock a function with NUMBER argument
    function = Function('lt', [parse('1')[0].parsed_selectors[0].pseudo_class.arguments[0]])
    result = translator.xpath_lt_function(xpath, function)
    assert 'position() < 2' in str(result)
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    function2 = Function('lt', [parse('5')[0].parsed_selectors[0].pseudo_class.arguments[0]])
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert 'position() < 6' in str(result2)
    
    # Test that it raises ExpressionError for non-number arguments
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(translator.xpathexpr_cls(), 
                                    Function('lt', [parse(':contains("test")')[0].parsed_selectors[0].pseudo_class.arguments[0]]))


# LLM-generated content at query #103
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'  # 2 + 1 = 3
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert result2.post_condition == 'position() < 1'  # 0 + 1 = 1
    
    # Test with negative value
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result3 = translator.xpath_lt_function(xpath3, function3)
    assert result3.post_condition == 'position() < 0'  # -1 + 1 = 0
    
    # Test that it raises ExpressionError for non-NUMBER argument types
    import pytest
    with pytest.raises(ExpressionError):
        xpath4 = translator.xpathexpr_cls()
        function4 = type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
        translator.xpath_lt_function(xpath4, function4)
```


# LLM-generated content at query #104
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with simple number argument
    xpath = translator.xpath_lt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '1'})()]
        })()
    )
    assert 'position() < 2' in str(xpath)
    
    # Test with zero value
    xpath = translator.xpath_lt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '0'})()]
        })()
    )
    assert 'position() < 1' in str(xpath)
    
    # Test with larger number
    xpath = translator.xpath_lt_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda: ['NUMBER'],
            'arguments': [type('Argument', (), {'value': '5'})()]
        })()
    )
    assert 'position() < 6' in str(xpath)
    
    # Test that it raises ExpressionError for non-number argument
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(
            translator.xpathexpr_cls(path='//h1', element='h1'),
            type('Function', (), {
                'argument_types': lambda: ['STRING'],
                'arguments': ['invalid']
            })()
        )


# LLM-generated content at query #105
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls(path='//div', element='div')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another number
    xpath2 = translator.xpathexpr_cls(path='//p', element='p')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with invalid argument type (should raise ExpressionError)
    xpath3 = translator.xpathexpr_cls(path='//span', element='span')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath3, function3)
```


# LLM-generated content at query #106
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with a simple numeric argument
    from cssselect.parser import Function, Number
    function = Function('gt', [Number('2')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    function2 = Function('gt', [Number('0')])
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 1'
    
    # Test that it raises ExpressionError for non-number arguments
    from cssselect.parser import Function, String
    xpath3 = translator.xpathexpr_cls()
    function3 = Function('gt', [String('invalid')])
    try:
        translator.xpath_gt_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #107
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, FakeFunction('NUMBER', '2'))
    assert result.post_condition == 'position() < 3'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, FakeFunction('NUMBER', '-1'))
    assert result.post_condition == 'position() < 0'
    
    # Test with zero
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_lt_function(xpath, FakeFunction('NUMBER', '0'))
    assert result.post_condition == 'position() < 1'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    try:
        translator.xpath_lt_function(xpath, FakeFunction('STRING', 'test'))
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt(), got" in str(e)
    
    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    try:
        translator.xpath_lt_function(xpath, FakeFunction('NUMBER', '1', '2'))
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt(), got" in str(e)


class FakeFunction:
    def __init__(self, *args):
        self.arguments = [FakeArgument(arg) for arg in args[1:]]
        self.argument_types_result = args[0] if len(args) == 1 else None
        
    def argument_types(self):
        if self.argument_types_result:
            return [self.argument_types_result]
        return [arg.type for arg in self.arguments]


class FakeArgument:
    def __init__(self, value, arg_type='NUMBER'):
        self.value = value
        self.type = arg_type
```


# LLM-generated content at query #108
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with a simple selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'), 
        MockFunction(['.bar'], 'STRING')
    )
    assert 'descendant::' in str(xpath)
    
    # Test with an ident argument
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        MockFunction(['div'], 'IDENT')
    )
    assert 'descendant::' in str(xpath)
    
    # Test that it raises ExpressionError for invalid argument types
    from cssselect.xpath import ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(element='div'),
            MockFunction(['1'], 'NUMBER')
        )
    
    # Test the post_condition is added correctly
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        MockFunction(['.bar'], 'STRING')
    )
    assert xpath.post_condition is not None
    assert 'descendant::' in xpath.post_condition
    
    # Test with a class selector
    xpath = translator.xpath_has_function(
        XPathExpr(element='div'),
        MockFunction(['.bar'], 'STRING')
    )
    assert 'contains' in str(xpath) or 'class' in str(xpath)

class MockFunction:
    """Helper class to mock Function objects from cssselect"""
    def __init__(self, arguments, argument_type):
        self.arguments = [self]
        self.arguments[0].value = arguments[0]
        self._argument_types = [argument_type]
        
    def argument_types(self):
        return self._argument_types
```


# LLM-generated content at query #109
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import Function, Token
    function = Function('gt', [Token('NUMBER', '0')])
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 1]'
    
    # Test with another number
    function = Function('gt', [Token('NUMBER', '2')])
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '*[position() > 3]'
    
    # Test with non-number argument should raise ExpressionError
    from cssselect.xpath import ExpressionError
    import pytest
    function = Function('gt', [Token('STRING', '"hello"')])
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, function)
    
    # Test with multiple arguments should raise ExpressionError
    function = Function('gt', [Token('NUMBER', '1'), Token('NUMBER', '2')])
    xpath = translator.xpathexpr_cls()
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_gt_function(xpath, function)
```


# LLM-generated content at query #110
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"
```


# LLM-generated content at query #111
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has functionality
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test with invalid argument type
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(path='//div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )
    
    # Test post_condition is properly added
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(path='//div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'test'})()]
        })()
    )
    assert xpath.post_condition is not None
    assert 'descendant::*' in str(xpath)


# LLM-generated content at query #112
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::*[contains(@class, "bar")]' in str(xpath) or 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with ID selector
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '#myid'})()]
        })()
    )
    assert '@id' in str(xpath)
    
    # Test error case - invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            XPathExpr(path='//div', element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
    
    # Test with IDENT argument type
    xpath = translator.xpath_has_function(
        XPathExpr(path='//div', element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'test'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
```


# LLM-generated content at query #113
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('arg', (), {'value': 'title'})()]
        })()
    )
    assert "contains(., 'title')" in str(xpath)
    
    # Test with ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(),
        type('function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('arg', (), {'value': 'content'})()]
        })()
    )
    assert "contains(., 'content')" in str(xpath)
    
    # Test with invalid argument types
    import pytest
    from cssselect.xpath import ExpressionError
    
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            type('function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('arg', (), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #114
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    from cssselect.parser import parse
    from cssselect.xpath import XPathExpr
    
    # Test basic usage: h1:gt(0) should give position() > 1
    xpath = XPathExpr()
    parsed = parse(':gt(0)')
    function = parsed[0].pseudo_class
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with different number: h1:gt(2) should give position() > 3
    xpath2 = XPathExpr()
    parsed2 = parse(':gt(2)')
    function2 = parsed2[0].pseudo_class
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test that it returns the xpath object
    assert result is xpath
    
    # Test with invalid argument type (should raise ExpressionError)
    parsed3 = parse(':contains("text")')
    function3 = parsed3[0].pseudo_class
    try:
        translator.xpath_gt_function(XPathExpr(), function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #115
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    
    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'test')"
    
    # Test with special characters in string
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': "it's"})()]
    })()
    result3 = translator.xpath_contains_function(xpath3, function3)
    assert result3.post_condition == "contains(., \"it's\")"
    
    # Test with multiple post_conditions (simulating chaining)
    xpath4 = XPathExpr()
    xpath4.add_post_condition('position() = 1')
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result4 = translator.xpath_contains_function(xpath4, function4)
    assert 'position() = 1' in result4.post_condition
    assert 'contains(., ' in result4.post_condition
```


# LLM-generated content at query #116
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Create a mock function object with NUMBER argument
    class MockArgument:
        def __init__(self, value):
            self.value = value
    
    class MockFunction:
        def __init__(self, value):
            self.arguments = [MockArgument(value)]
        
        def argument_types(self):
            return ['NUMBER']
    
    # Test with value 0 (should add condition 'position() > 1')
    xpath = XPathExpr()
    function = MockFunction(0)
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    assert result is xpath
    
    # Test with value 2 (should add condition 'position() > 3')
    xpath2 = XPathExpr()
    function2 = MockFunction(2)
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative value -1 (should add condition 'position() > 0')
    xpath3 = XPathExpr()
    function3 = MockFunction(-1)
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test with value 5 (should add condition 'position() > 6')
    xpath4 = XPathExpr()
    function4 = MockFunction(5)
    result4 = translator.xpath_gt_function(xpath4, function4)
    assert result4.post_condition == 'position() > 6'
    
    # Test that ExpressionError is raised for non-NUMBER argument types
    class NonNumberFunction:
        def __init__(self):
            self.arguments = [MockArgument("string")]
        
        def argument_types(self):
            return ['STRING']
    
    xpath5 = XPathExpr()
    function5 = NonNumberFunction()
    try:
        translator.xpath_gt_function(xpath5, function5)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that ExpressionError is raised for multiple arguments
    class MultipleArgsFunction:
        def __init__(self):
            self.arguments = [MockArgument(1), MockArgument(2)]
        
        def argument_types(self):
            return ['NUMBER', 'NUMBER']
    
    xpath6 = XPathExpr()
    function6 = MultipleArgsFunction()
    try:
        translator.xpath_gt_function(xpath6, function6)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with empty arguments
    class EmptyArgsFunction:
        def __init__(self):
            self.arguments = []
        
        def argument_types(self):
            return []
    
    xpath7 = XPathExpr()
    function7 = EmptyArgsFunction()
    try:
        translator.xpath_gt_function(xpath7, function7)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that post_condition is added correctly when there's an existing post_condition
    xpath8 = XPathExpr()
    xpath8.post_condition = 'position() > 0'
    function8 = MockFunction(1)
    result8 = translator.xpath_gt_function(xpath8, function8)
    assert result8.post_condition == 'position() > 0 and (position() > 2)'
```


# LLM-generated content at query #117
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid NUMBER argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with index 0
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 1'
    
    # Test with invalid argument type (not NUMBER)
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    
    try:
        translator.xpath_gt_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt()" in str(e)
```


# LLM-generated content at query #118
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '2'})()]
    
    result = translator.xpath_gt_function(xpath, MockFunction())
    assert result.post_condition == 'position() > 3', "Should add position > 3 for gt(2)"
    
    # Test with value 0
    xpath2 = translator.xpathexpr_cls()
    class MockFunctionZero:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('MockArgument', (), {'value': '0'})()]
    
    result2 = translator.xpath_gt_function(xpath2, MockFunctionZero())
    assert result2.post_condition == 'position() > 1', "Should add position > 1 for gt(0)"
    
    # Test that it raises ExpressionError for non-number arguments
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), MockFunctionInvalid())
```


# LLM-generated content at query #119
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'text'})()]
    })()
    
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath3, function3)
    assert "Expected a single string or ident" in str(excinfo.value)
    
    # Test with multiple arguments
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [type('Argument', (), {'value': 'a'}), type('Argument', (), {'value': 'b'})()]
    })()
    
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath4, function4)
    assert "Expected a single string or ident" in str(excinfo.value)
    
    # Test with special characters in string
    xpath5 = XPathExpr()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': "it's"})()]
    })()
    
    result5 = translator.xpath_contains_function(xpath5, function5)
    assert result5.post_condition == "contains(., \"it's\")"
    
    # Test with empty string
    xpath6 = XPathExpr()
    function6 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': ''})()]
    })()
    
    result6 = translator.xpath_contains_function(xpath6, function6)
    assert result6.post_condition == "contains(., '')"


# LLM-generated content at query #120
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    from cssselect.parser import Function, parse
    func = Function('contains', [parse('"title"')])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with ident argument
    func = Function('contains', [parse('title')])
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls(), func)
    assert 'contains(., "title")' in str(xpath)
    
    # Test with invalid argument type
    import pytest
    from cssselect.xpath import ExpressionError
    func = Function('contains', [parse('123')])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(translator.xpathexpr_cls(), func)


# LLM-generated content at query #121
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test 1: Basic has selector matching child class
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': '.baz'})()]
        })()
    )
    assert 'descendant::*[contains(@class, "baz")]' in str(xpath)
    
    # Test 2: Has selector with tag name
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'div'})()]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test 3: Has selector with IDENT type
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'span'})()]
        })()
    )
    assert 'descendant::span' in str(xpath)


# LLM-generated content at query #122
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)
    
    # Test has with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': 'div'})]
        })()
    )
    assert 'descendant::div' in str(xpath)
    
    # Test has with no matches
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.baz'})]
        })()
    )
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]' in str(xpath)
    
    # Test error case - wrong argument types
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '0'})]
            })()
        )


# LLM-generated content at query #123
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    
    # Test with a mock function that has NUMBER argument type
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Arg', (), {'value': '2'})()]
    
    # Create a mock XPath expression
    xpath = XPathExpr('div', 'div', '')
    
    result = translator.xpath_eq_function(xpath, MockFunction())
    assert result.post_condition == 'position() = 3'
    
    # Test with first element (index 0)
    class MockFunctionFirst:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('Arg', (), {'value': '0'})()]
    
    xpath2 = XPathExpr('div', 'div', '')
    result2 = translator.xpath_eq_function(xpath2, MockFunctionFirst())
    assert result2.post_condition == 'position() = 1'
    
    # Test with non-NUMBER argument type should raise ExpressionError
    class MockFunctionInvalid:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Arg', (), {'value': 'test'})()]
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(XPathExpr('div', 'div', ''), MockFunctionInvalid())
```


# LLM-generated content at query #124
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = XPathExpr(path='//h1', element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath = XPathExpr(path='//h1', element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test with invalid argument type
    xpath = XPathExpr(path='//h1', element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with special characters in text
    xpath = XPathExpr(path='//h1', element='h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': "it's"})()]
    })()
    
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., \"it's\")"
```


# LLM-generated content at query #125
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    assert result.path == ''
    assert result.element == '*'
    assert result.condition == ''


# LLM-generated content at query #126
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test basic contains with string
    xpath = XPathExpr('//h1', 'h1')
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'title'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'title')"
    
    # Test basic contains with ident
    xpath2 = XPathExpr('//div', 'div')
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'content'})()]
    })()
    result2 = translator.xpath_contains_function(xpath2, function2)
    assert result2.post_condition == "contains(., 'content')"
    
    # Test that it raises ExpressionError for invalid argument types
    xpath3 = XPathExpr('//span', 'span')
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath3, function3)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test that it raises ExpressionError for empty arguments
    xpath4 = XPathExpr('//p', 'p')
    function4 = type('Function', (), {
        'argument_types': lambda self: [],
        'arguments': []
    })()
    try:
        translator.xpath_contains_function(xpath4, function4)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
    
    # Test with special characters in text
    xpath5 = XPathExpr('//a', 'a')
    function5 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': "it's"})()]
    })()
    result5 = translator.xpath_contains_function(xpath5, function5)
    assert result5.post_condition == "contains(., \"it's\")" or result5.post_condition == "contains(., 'it\\'s')"


# LLM-generated content at query #127
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test basic has selector with string argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Argument', (), {'value': '.bar'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'bar' in str(xpath)
    
    # Test has selector with ident argument
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(element='div'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Argument', (), {'value': 'span'})()]
        })()
    )
    assert 'descendant::' in str(xpath)
    assert 'span' in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(element='div'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Argument', (), {'value': '1'})()]
            })()
        )
```


# LLM-generated content at query #128
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
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '3'})()]
    })()
    result2 = translator.xpath_eq_function(xpath2, function2)
    assert result2.post_condition == 'position() = 4'
    
    # Test that invalid argument type raises ExpressionError
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    try:
        translator.xpath_eq_function(xpath3, function3)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
    
    # Test that multiple arguments raises ExpressionError
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'}), type('Arg', (), {'value': '2'})()]
    })()
    try:
        translator.xpath_eq_function(xpath4, function4)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #129
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument
    xpath = translator.xpathexpr_cls(path='/', element='*')
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('obj', (object,), {'value': 'title'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunction())
    assert result.post_condition == "contains(., 'title')"
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls(path='/', element='*')
    class MockFunctionIdent:
        def argument_types(self):
            return ['IDENT']
        arguments = [type('obj', (object,), {'value': 'text'})()]
    
    result = translator.xpath_contains_function(xpath, MockFunctionIdent())
    assert result.post_condition == "contains(., 'text')"
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls(path='/', element='*')
    class MockFunctionInvalid:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('obj', (object,), {'value': '42'})()]
    
    import pytest
    from cssselect.xpath import ExpressionError
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, MockFunctionInvalid())


# LLM-generated content at query #130
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with a simple xpath
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, mock_function)
    assert str(result) == '*[position() > 1]'
    
    # Test with value 1
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, mock_function)
    assert str(result) == '*[position() > 2]'
    
    # Test with negative value
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, mock_function)
    assert str(result) == '*[position() > 0]'
    
    # Test with element type
    xpath = translator.xpathexpr_cls()
    xpath.element = 'div'
    mock_function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    
    result = translator.xpath_gt_function(xpath, mock_function)
    assert str(result) == 'div[position() > 3]'
    
    # Test that exception is raised for non-NUMBER argument type
    xpath = translator.xpathexpr_cls()
    mock_function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, mock_function)
```


# LLM-generated content at query #131
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with STRING argument type
    result = translator.xpath_contains_function(xpath, MagicMock(
        argument_types=lambda: ['STRING'],
        arguments=[MagicMock(value='test text')]
    ))
    assert 'contains(., "test text")' in str(result)
    
    # Test with IDENT argument type
    xpath2 = translator.xpathexpr_cls()
    result2 = translator.xpath_contains_function(xpath2, MagicMock(
        argument_types=lambda: ['IDENT'],
        arguments=[MagicMock(value='test_ident')]
    ))
    assert 'contains(., "test_ident")' in str(result2)
    
    # Test that it raises ExpressionError for invalid argument types
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(),
            MagicMock(
                argument_types=lambda: ['NUMBER'],
                arguments=[MagicMock(value=1)]
            )
        )


# LLM-generated content at query #132
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    
    # Test with valid number argument
    from cssselect.parser import Function, Token
    function = Function('lt', [Token('NUMBER', '2')])
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '*[position() < 3]'
    
    # Test with different number
    xpath2 = translator.xpathexpr_cls()
    function2 = Function('lt', [Token('NUMBER', '0')])
    result2 = translator.xpath_lt_function(xpath2, function2)
    assert str(result2) == '*[position() < 1]'
    
    # Test with invalid argument type
    from cssselect.xpath import ExpressionError
    import pytest
    xpath3 = translator.xpathexpr_cls()
    function3 = Function('lt', [Token('STRING', 'invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath3, function3)
    
    # Test with multiple arguments
    xpath4 = translator.xpathexpr_cls()
    function4 = Function('lt', [Token('NUMBER', '1'), Token('NUMBER', '2')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath4, function4)
```


# LLM-generated content at query #133
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test basic contains with string argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['STRING'],
            'arguments': [type('Arg', (), {'value': 'title'})()]
        })()
    )
    assert 'contains' in str(xpath)
    assert "'title'" in str(xpath)
    
    # Test contains with ident argument
    xpath = translator.xpath_contains_function(
        translator.xpathexpr_cls(path='//h1', element='h1'),
        type('Function', (), {
            'argument_types': lambda self: ['IDENT'],
            'arguments': [type('Arg', (), {'value': 'content'})()]
        })()
    )
    assert 'contains' in str(xpath)
    assert "'content'" in str(xpath)
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(
            translator.xpathexpr_cls(path='//h1', element='h1'),
            type('Function', (), {
                'argument_types': lambda self: ['NUMBER'],
                'arguments': [type('Arg', (), {'value': '1'})()]
            })()
        )


# LLM-generated content at query #134
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'
    
    # Test with value 0
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'
    
    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'
    
    # Test that it raises ExpressionError for invalid argument types
    xpath = translator.xpathexpr_cls()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    
    with pytest.raises(ExpressionError, match="Expected a single integer for :gt"):
        translator.xpath_lt_function(xpath, function)
```


# LLM-generated content at query #135
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with a simple selector string
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert 'descendant::' in result.post_condition
    assert 'bar' in result.post_condition
    
    # Test with an ident
    xpath2 = XPathExpr()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result2 = translator.xpath_has_function(xpath2, function2)
    assert 'descendant::' in result2.post_condition
    assert 'div' in result2.post_condition
    
    # Test that invalid argument types raise ExpressionError
    import pytest
    from cssselect.xpath import ExpressionError
    xpath3 = XPathExpr()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath3, function3)
    
    # Test with a more complex selector
    xpath4 = XPathExpr()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.foo > .bar'})()]
    })()
    result4 = translator.xpath_has_function(xpath4, function4)
    assert 'descendant::' in result4.post_condition
    assert 'foo' in result4.post_condition
    assert 'bar' in result4.post_condition
```


# LLM-generated content at query #136
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = Mock(function)
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='2')]
    
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 3'
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function.argument_types.return_value = ['STRING']
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #137
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    
    # Test with STRING argument type
    mock_xpath = XPathExpr()
    mock_function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test text'})()]
    })()
    
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert "contains(., 'test text')" in result.post_condition
    
    # Test with IDENT argument type
    mock_xpath2 = XPathExpr()
    mock_function2 = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test_ident'})()]
    })()
    
    result2 = translator.xpath_contains_function(mock_xpath2, mock_function2)
    assert "contains(., 'test_ident')" in result2.post_condition
    
    # Test with invalid argument type should raise ExpressionError
    mock_xpath3 = XPathExpr()
    mock_function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '42'})()]
    })()
    
    import pytest
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(mock_xpath3, mock_function3)


# LLM-generated content at query #138
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
    assert str(result) == '*[position() > 1]'
    
    # Test with index 1
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert str(result2) == '*[position() > 2]'
    
    # Test with higher index
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '5'})()]
    })()
    
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert str(result3) == '*[position() > 6]'
    
    # Test that it raises ExpressionError for non-NUMBER arguments
    import pytest
    from cssselect.xpath import ExpressionError
    
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(translator.xpathexpr_cls(), function4)
```


# LLM-generated content at query #139
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    
    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant-or-self::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"
    
    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::div"
    
    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    import pytest
    with pytest.raises(ExpressionError, match="Expected a single string or ident for :has"):
        translator.xpath_has_function(xpath, function)
    
    # Test that post_condition is properly set
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.test'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition is not None
    assert isinstance(result.post_condition, str)


# LLM-generated content at query #140
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    
    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'
    
    # Test with another valid number
    xpath2 = translator.xpathexpr_cls()
    function2 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result2 = translator.xpath_gt_function(xpath2, function2)
    assert result2.post_condition == 'position() > 3'
    
    # Test with negative number
    xpath3 = translator.xpathexpr_cls()
    function3 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result3 = translator.xpath_gt_function(xpath3, function3)
    assert result3.post_condition == 'position() > 0'
    
    # Test with invalid argument type (should raise ExpressionError)
    import pytest
    from cssselect.xpath import ExpressionError
    xpath4 = translator.xpathexpr_cls()
    function4 = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath4, function4)
    
    # Test with multiple arguments (should raise ExpressionError)
    xpath5 = translator.xpathexpr_cls()
    function5 = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath5, function5)
```


