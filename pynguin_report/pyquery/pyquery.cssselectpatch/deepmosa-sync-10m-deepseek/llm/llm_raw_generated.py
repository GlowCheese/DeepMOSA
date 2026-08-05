####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_xpath_lt_function_positive_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_negative_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '-1'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 0']

def test_xpath_lt_function_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']
```


# LLM-generated content at query #2
#--------------------------

```
def test_xpath_input_pseudo_adds_condition_for_input_select_textarea_button():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_input_pseudo(xpath)
    assert xpath.path == 'test'
    assert xpath.condition == "(name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button')"
    assert xpath.post_conditions == []


# LLM-generated content at query #3
#--------------------------

```
def test_xpath_even_pseudo_adds_correct_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_even_pseudo(xpath)
    assert result.post_conditions == ['position() mod 2 = 1']

def test_xpath_even_pseudo_returns_xpath():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_even_pseudo(xpath)
    assert result is xpath
```


# LLM-generated content at query #4
#--------------------------

```
def test_xpath_even_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_even_pseudo(xpath)
    assert result.condition == 'position() mod 2 = 1' if hasattr(result, 'condition') else True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert translator.xpathexpr_cls == XPathExpr
```


# LLM-generated content at query #6
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='title')]
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = MagicMock()
    function.argument_types.return_value = ['IDENT']
    function.arguments = [MagicMock(value='text')]
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'text')"]

def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value=1)]
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #7
#--------------------------

```
def test_xpath_gt_function_positive_index():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 3']

def test_xpath_gt_function_zero_index():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']

def test_xpath_gt_function_negative_index():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 0']

def test_xpath_gt_function_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='text')]
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #8
#--------------------------

```
def test_xpath_eq_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_second_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '1'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 2']

def test_xpath_eq_function_raises_error_for_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': 'test'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except Exception as e:
        assert 'Expected a single integer for :eq()' in str(e)
```


# LLM-generated content at query #9
#--------------------------

```
def test_lower_case_attribute_names_true():
    translator = JQueryTranslator(xhtml=False)
    assert translator.lower_case_attribute_names == True
```


# LLM-generated content at query #10
#--------------------------

```
def test_xpath_gt_function_with_non_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from cssselect.parser import Function
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [('STRING', 'not_a_number')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except Exception as e:
        assert str(e) == "Expected a single integer for :gt(), got [('STRING', 'not_a_number')]"
```


# LLM-generated content at query #11
#--------------------------

```
def test_lt_function_raises_error_for_non_number_argument():
    from pyquery.jquery_translator import JQueryTranslator, ExpressionError
    from cssselect.parser import Function
    translator = JQueryTranslator()
    xpath = translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(translator.xpath_eq_function(


# LLM-generated content at query #12
#--------------------------

```
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result is not None

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result is not None

def test_xpath_contains_function_raises_on_invalid_args():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_contains_function_raises_on_empty_args():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: [], 'arguments': []})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #13
#--------------------------

```
def test_lower_case_attribute_names_false_when_xhtml_true():
    from pyquery.jquerytranslator import JQueryTranslator
    translator = JQueryTranslator(xhtml=True)
    assert not translator.lower_case_attribute_names
```


# LLM-generated content at query #14
#--------------------------

```
def test_xpath_lt_function_raises_on_non_number():
    mock_function = type('MockFunction', (), {'argument_types': lambda self: ['STRING'], 'arguments': []})()
    translator = JQueryTranslator()
    try:
        translator.xpath_lt_function(None, mock_function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #15
#--------------------------

```
def test_xpath_has_function_with_matching_selector():
    from pyquery.translator import JQueryTranslator
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath(), type('function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('arg', (object,), {'value': '.bar'})]})())
    assert ".bar" in str(xpath)

def test_xpath_has_function_with_non_matching_selector():
    from pyquery.translator import JQueryTranslator
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath(), type('function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('arg', (object,), {'value': '.baz'})]})())
    assert ".baz" in str(xpath)

def test_xpath_has_function_with_ident_argument():
    from pyquery.translator import JQueryTranslator
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath(), type('function', (object,), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('arg', (object,), {'value': 'div'})]})())
    assert "descendant::div" in str(xpath)

def test_xpath_has_function_invalid_argument_type():
    from pyquery.translator import JQueryTranslator
    from pyquery.expression import ExpressionError
    translator = JQueryTranslator()
    try:
        translator.xpath_has_function(translator.xpath(), type('function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('arg', (object,), {'value': '1'})]})())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #16
#--------------------------

```
def test_xpath_lt_function_raises_error_when_argument_types_not_number():
    function = type('FakeFunction', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('FakeArg', (), {'value': 'test'})()]})()
    xpath = type('FakeXPath', (), {'add_post_condition': lambda self, cond: None})()
    translator = JQueryTranslator()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        assert True
```


# LLM-generated content at query #17
#--------------------------

```
def test_xpath_lt_function_non_number_raises_expression_error():
    from pyquery.translator import JQueryTranslator
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [Token('STRING', 'test')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #18
#--------------------------

```python
def test_xpath_has_function_valid_argument_type_string():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '.bar'
    translator.css_to_xpath = MagicMock(return_value='descendant::*[contains(concat(" ", @class, " "), " bar ")]')
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #19
#--------------------------

```
def test_xpath_contains_function_raises_for_non_string_or_ident():
    translator = JQueryTranslator()
    function = MockFunction()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MockArgument()]
    function.arguments[0].value = '42'
    xpath = XPathExpr()
    raised = False
    try:
        translator.xpath_contains_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_eq_function_uses_number_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3'] or result is xpath
```


# LLM-generated content at query #21
#--------------------------

def test_xpath_contains_function_raises_for_non_string_or_ident_args():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['NUMBER'], arguments=[MockArgument(value='42')])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #22
#--------------------------

```
def test_xpath_eq_function_valid_number():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Numeric
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Numeric('0', 'NUMBER')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_second_element():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Numeric
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Numeric('1', 'NUMBER')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 2']

def test_xpath_eq_function_negative_index():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Numeric
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Numeric('-1', 'NUMBER')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 0']

def test_xpath_eq_function_invalid_argument_type():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Function
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Function('foo', [])])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except Exception:
        pass
```


# LLM-generated content at query #23
#--------------------------

```
def test_xpath_has_function_matching_selector_returns_elements():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".bar")')
    assert len(result) == 1
    assert result[0].get('class') == 'foo'

def test_xpath_has_function_non_matching_selector_returns_empty():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".baz")')
    assert len(result) == 0

def test_xpath_has_function_matching_tag_selector_returns_elements():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(div)')
    assert len(result) == 1
    assert result[0].get('class') == 'foo'

def test_xpath_has_function_self_matching_selector_returns_empty():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".foo")')
    assert len(result) == 0

def test_xpath_has_function_raises_error_for_non_string_ident_argument():
    from pyquery.translator import JQueryTranslator
    from unittest.mock import Mock
    translator = JQueryTranslator()
    xpath = Mock()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='42')]
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert 'Expected a single string or ident for :has()' in str(e)
```


# LLM-generated content at query #24
#--------------------------

```
def test_xpath_lt_function_with_number_argument_does_not_raise():
    from pyquery.jquery_translator import JQueryTranslator
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #25
#--------------------------

```
def test_xpath_gt_function_argument_types_is_number():
    from pyquery.jquery_translator import JQueryTranslator, XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result is not None
```


# LLM-generated content at query #26
#--------------------------

```python
def test_xpath_contains_function_valid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='test')]
    translator.xpath_literal = MagicMock(return_value="'test'")
    result = translator.xpath_contains_function(xpath, function)
    assert result == xpath
```


# LLM-generated content at query #27
#--------------------------

```python
def test_xpath_eq_function_accepts_number_argument(self):
    translator = JQueryTranslator()
    xpath = translator.xpath_eq_function(
        XPathExpr('test'),
        MockFunction(['NUMBER'])
    )
    assert xpath is not None
```


# LLM-generated content at query #28
#--------------------------

```
def test_xpath_gt_function_with_non_number_argument():
    from pyquery.translator import JQueryTranslator
    from cssselect.parser import Function
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [('STRING', 'abc')])
    try:
        translator.xpath_gt_function(xpath, function)
        raised = False
    except Exception:
        raised = True
    assert raised
```


# LLM-generated content at query #29
#--------------------------

```
def test_xpath_eq_function_raises_for_non_number_arguments():
    from pyquery.jquery_translator import JQueryTranslator
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('eq', [Token('STRING', 'not_a_number')])
    raised = False
    try:
        translator.xpath_eq_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #30
#--------------------------

def test_xpath_lt_function_argument_types_is_number():
    from pyquery import PyQuery
    d = PyQuery('<div><h1 class="first"/><h1 class="last"/></div>')
    result = d('h1:lt(1)')
    assert len(result) == 1
    assert result[0].get('class') == 'first'


# LLM-generated content at query #31
#--------------------------

```
def test_xpath_gt_function_raises_for_non_number_arguments():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath import XPathExpr
    from lxml.cssselect import ExpressionError, FunctionElement
    translator = JQueryTranslator()
    xpath = XPathExpr()
    token = type('Token', (), {'value': 'string', 'type': 'STRING'})()
    function = FunctionElement('gt', [token])
    raised = False
    try:
        translator.xpath_gt_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #32
#--------------------------

```
def test_xpath_has_function_returns_xpath_with_condition():
    from pyquery.pyquery import PyQuery
    from pyquery.css_selectors import JQueryTranslator
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath('div'), type('function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('arg', (), {'value': '.bar'})]})())
    assert xpath is not None
```


# LLM-generated content at query #33
#--------------------------

```
def test_xpath_eq_function_returns_correct_xpath_for_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Function("eq", [Number("0")])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_returns_correct_xpath_for_positive_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Function("eq", [Number("5")])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 6']

def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Function("eq", [String("foo")])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_eq_function_raises_error_for_multiple_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Function("eq", [Number("1"), Number("2")])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #34
#--------------------------

```python
def test_xpath_has_function_valid_argument_type_string():
    translator = JQueryTranslator()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '.bar'
    xpath = MagicMock()
    translator.xpath_has_function(xpath, function)
    assert xpath.add_post_condition.called
```


# LLM-generated content at query #35
#--------------------------

```
def test_xpath_lt_function_raises_expression_error_for_non_number():
    from pyquery.translator import JQueryTranslator, ExpressionError
    from cssselect.parser import Function
    translator = JQueryTranslator()
    xpath = translator.xpath_lt_function(translator.xpathexpr_cls(), Function('lt', [('STRING', '"foo"')]))
    assert False, "Should have raised ExpressionError"
```


# LLM-generated content at query #36
#--------------------------

```
def test_xpath_eq_function_returns_correct_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']


# LLM-generated content at query #37
#--------------------------

```
def test_xpath_has_function_returns_xpath_with_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath

def test_xpath_has_function_raises_on_invalid_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': 1})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_has_function_works_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert result.post_conditions

def test_xpath_has_function_works_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'div'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert result.post_conditions
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_xpath_submit_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_submit_pseudo(xpath)
    assert result is xpath
    assert "@type = 'submit' and name(.) = 'input'" in str(xpath)
```


# LLM-generated content at query #2
#--------------------------

```python
from pyquery.pyquery import PyQuery
from pyquery.jquery_translator import JQueryTranslator

def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert translator.xpathexpr_cls is not None
    assert translator.xpathexpr_cls.__name__ == 'XPathExpr'
```


# LLM-generated content at query #3
#--------------------------

```
def test_xpath_disabled_pseudo_adds_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result is xpath
    condition = result.condition
    assert 'disabled' in condition
    assert 'and' in condition
    assert 'or' in condition
    assert 'not' not in condition

def test_xpath_disabled_pseudo_condition_structure():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert xpath.condition == '''(
            ((name(.) = 'button' or name(.) = 'input' or name(.) = 'select'
                    or name(.) = 'textarea' or name(.) = 'fieldset')
                and (@disabled or (ancestor::fieldset[@disabled]
                    and not(ancestor::legend[not(preceding-sibling::legend)])))
            )
            or
            ((name(.) = 'option'
                and (@disabled or ancestor::optgroup[@disabled]))
            )
            or
            ((name(.) = 'optgroup' and (@disabled)))
            )''' 
```


# LLM-generated content at query #4
#--------------------------

```
def test_xpath_disabled_pseudo_returns_self():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result is xpath

def test_xpath_disabled_pseudo_adds_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert xpath.condition is not None

def test_xpath_disabled_pseudo_includes_disabled_attribute():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert '@disabled' in xpath.condition

def test_xpath_disabled_pseudo_includes_button():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'button'" in xpath.condition

def test_xpath_disabled_pseudo_includes_input():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'input'" in xpath.condition

def test_xpath_disabled_pseudo_includes_select():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'select'" in xpath.condition

def test_xpath_disabled_pseudo_includes_textarea():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'textarea'" in xpath.condition

def test_xpath_disabled_pseudo_includes_fieldset():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'fieldset'" in xpath.condition

def test_xpath_disabled_pseudo_includes_option():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'option'" in xpath.condition

def test_xpath_disabled_pseudo_includes_optgroup():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    translator.xpath_disabled_pseudo(xpath)
    assert "name(.) = 'optgroup'" in xpath.condition
```


# LLM-generated content at query #5
#--------------------------

```python
def test_xpath_disabled_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//*')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.path == '//*'
    assert 'disabled' in result.condition


# LLM-generated content at query #6
#--------------------------

```
def test_xpath_gt_function_returns_xpath_with_position_gt_condition():
    from pyquery.translator import JQueryTranslator, XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 3']

def test_xpath_gt_function_zero_index():
    from pyquery.translator import JQueryTranslator, XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']

def test_xpath_gt_function_negative_raises_expression_error():
    from pyquery.translator import JQueryTranslator, XPathExpr
    from pyquery.translator import ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'abc'})()]})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_gt_function_returns_same_xpath_object():
    from pyquery.translator import JQueryTranslator, XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #7
#--------------------------

```
def test_xpath_has_function_returns_xpath_with_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': '.bar'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath

def test_xpath_has_function_raises_error_for_non_string_or_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '1'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert "Expected a single string or ident for :has()" in str(e)

def test_xpath_has_function_adds_correct_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('div')
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': '.bar'})]})()
    translator.xpath_has_function(xpath, function)
    assert "descendant::" in xpath.post_conditions[0]


# LLM-generated content at query #8
#--------------------------

```
def test_xpath_lt_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    from cssselect.parser import Function, Token
    function = Function('lt', [Token('NUMBER', '2')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_raises_error_for_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    from cssselect.parser import Function, Token
    function = Function('lt', [Token('STRING', 'abc')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_lt_function_raises_error_for_empty_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    from cssselect.parser import Function
    function = Function('lt', [])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_has_function_with_string_argument_type():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.pyquery import PyQuery
    from pyquery.cssselect import XPathExpr
    from pyquery.cssselect import Function
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [type('arg', (), {'value': '.bar', 'type': 'STRING'})()])
    function.argument_types = lambda: ['STRING']
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #10
#--------------------------

```
def test_xpath_gt_function_raises_expression_error_for_non_number_argument():
    from pyquery.css_selectparser import JQueryTranslator
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = translator.xpath_gt_function(None, Function('gt', [Token('IDENT', 'foo')]))
    # This will raise ExpressionError if argument_types is not ['NUMBER']
```


# LLM-generated content at query #11
#--------------------------

```
def test_xpath_has_function_raises_on_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['NUMBER'])
    function.arguments = [MockArgument(value='0')]
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #12
#--------------------------

```
def test_xpath_gt_function_raises_on_non_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator, ExpressionError
    from pyquery.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #13
#--------------------------

```
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']
```


# LLM-generated content at query #14
#--------------------------

```
def test_xpath_has_function_returns_element_when_selector_matches_descendant():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_function("has"), type("function", (), {"argument_types": lambda self: ["STRING"], "arguments": [type("arg", (), {"value": ".bar"})]})())
    assert xpath.post_conditions[0] == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

def test_xpath_has_function_returns_empty_when_selector_does_not_match():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_function("has"), type("function", (), {"argument_types": lambda self: ["STRING"], "arguments": [type("arg", (), {"value": ".baz"})]})())
    assert xpath.post_conditions[0] == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

def test_xpath_has_function_accepts_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_function("has"), type("function", (), {"argument_types": lambda self: ["IDENT"], "arguments": [type("arg", (), {"value": "div"})]})())
    assert xpath.post_conditions[0] == "descendant::div"
```


# LLM-generated content at query #15
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_contains_function(translator.xpath_eq_function(translator.xpath_first_pseudo(translator.xpath_expr_cls()), None), None)

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_contains_function(translator.xpath_expr_cls(), None)

def test_xpath_contains_function_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpath_contains_function(translator.xpath_expr_cls(), None)
```


# LLM-generated content at query #16
#--------------------------

```
def test_xpath_lt_function_returns_xpath_with_correct_position_condition():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['NUMBER'], [NumberMock(2)])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['STRING'], [StringMock('test')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_lt_function_raises_error_for_multiple_arguments():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['NUMBER', 'NUMBER'], [NumberMock(1), NumberMock(2)])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_lt_function_returns_same_xpath_instance():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['NUMBER'], [NumberMock(0)])
    result = translator.xpath_lt_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #17
#--------------------------

```
def test_xpath_eq_function_returns_correct_xpath_for_first_element():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ["position() = 1"]

def test_xpath_eq_function_returns_correct_xpath_for_second_element():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ["position() = 2"]

def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert "Expected a single integer for :eq()" in str(e)
```


# LLM-generated content at query #18
#--------------------------

```
def test_xpath_eq_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3']

def test_xpath_eq_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'abc'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_eq_function_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_negative_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '-1'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 0']
```


# LLM-generated content at query #19
#--------------------------

```
def test_xpath_lt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(['STRING'])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #20
#--------------------------

def test_xpath_lt_function_argument_types_is_number():
    from pyquery.translator import JQueryTranslator, XPathExpr
    from cssselect.xpath import ExpressionError
    class MockFunction:
        def argument_types(self):
            return ['NUMBER']
        arguments = [type('arg', (), {'value': '2'})()]
    translator = JQueryTranslator()
    xpath = XPathExpr()
    result = translator.xpath_lt_function(xpath, MockFunction())
    assert result is xpath


# LLM-generated content at query #21
#--------------------------

```
def test_xpath_contains_function_raises_error_with_non_string_or_ident_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    try:
        translator.xpath_contains_function(None, Function('contains', [Token('NUMBER', '123')]))
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert type(e).__name__ == 'ExpressionError'
```


# LLM-generated content at query #22
#--------------------------

```
def test_xpath_has_function_with_valid_string_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #23
#--------------------------

```
def test_xpath_lt_function_with_non_number_argument_raises_expression_error():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    token = Token('STRING', 'not a number')
    function = Function('lt', [token])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #24
#--------------------------

```
def test_xpath_eq_function_with_non_number_argument():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath_expr import XPathExpr
    from cssselect.parser import FunctionalPseudoElement, Token, parse
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionalPseudoElement('eq', [Token('STRING', 'not_a_number')])
    raised = False
    try:
        translator.xpath_eq_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #25
#--------------------------

```
def test_xpath_lt_function_raises_error_for_non_number_arg():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.jquerytranslator import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #26
#--------------------------

```
def test_xpath_lt_function_non_number_raises_expression_error():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=[MockArgument(value='abc')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #27
#--------------------------

```
def test_xpath_eq_function_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Argument', (object,), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']


# LLM-generated content at query #28
#--------------------------

```
def test_xpath_has_function_raises_error_for_number_argument(self):
    translator = JQueryTranslator()
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['NUMBER']
    mock_function.arguments = [MagicMock(value='123')]
    mock_xpath = MagicMock()
    with self.assertRaises(ExpressionError):
        translator.xpath_has_function(mock_xpath, mock_function)
```


# LLM-generated content at query #29
#--------------------------

```
def test_xpath_eq_function_with_valid_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3']

def test_xpath_eq_function_with_first_element():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_non_number_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_eq_function_returns_xpath():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '5'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result == xpath
```


# LLM-generated content at query #30
#--------------------------

```
def test_xpath_eq_function_accepts_number():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath_expr import XPathExpr
    from cssselect.parser import Function
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [('NUMBER', '0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result is xpath
    assert result.post_conditions == ['position() = 1']
```


# LLM-generated content at query #31
#--------------------------

```python
from pyquery.pyquery import PyQuery
from pyquery.css_jquery import JQueryTranslator
from cssselect.parser import Function, parse

def test_xpath_has_function_raises_error_for_non_string_ident_args():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function.__wrapped__ if hasattr(translator.xpath_has_function, '__wrapped__') else translator.xpath_has_function
    dummy_xpath = type('XPath', (object,), {'add_post_condition': lambda self, x: None})()
    dummy_function = Function('has', [type('Arg', (object,), {'value': '1', 'type': 'NUMBER'})()])
    try:
        translator.xpath_has_function(dummy_xpath, dummy_function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        assert True
    except Exception as e:
        assert False, f"Unexpected exception: {e}"```


# LLM-generated content at query #32
#--------------------------

```
def test_xpath_gt_function_raises_on_non_number():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.pyquery import PyQuery
    from lxml.cssselect import ExpressionError
    translator = JQueryTranslator()
    function = type('Function', (object,), {'arguments': [type('Arg', (object,), {'value': 'string'})()],
                                            'argument_types': lambda self: ['STRING']})()
    xpath = type('XPath', (object,), {'add_post_condition': lambda self, cond: None})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #33
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'title'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result == xpath

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'content'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result == xpath

def test_xpath_contains_function_raises_error_on_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #34
#--------------------------

```
def test_xpath_contains_function_raises_expression_error_when_argument_types_not_string_or_ident():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('MockFunction', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('MockArg', (), {'value': '42'})]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #35
#--------------------------

```
def test_xpath_eq_function_raises_error_when_argument_types_not_number():
    mock_function = type('MockFunction', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('MockArgument', (), {'value': 'test'})]})()
    translator = JQueryTranslator()
    mock_xpath = type('MockXPath', (), {'add_post_condition': lambda self, condition: None})()
    try:
        translator.xpath_eq_function(mock_xpath, mock_function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #36
#--------------------------

def test_xpath_contains_function_invalid_argument_type_raises_expression_error():
    translator = JQueryTranslator()
    mock_function = type('MockFunction', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('MockArgument', (), {'value': '42'})()]})()
    mock_xpath = type('MockXPath', (), {'add_post_condition': lambda self, cond: None})()
    try:
        translator.xpath_contains_function(mock_xpath, mock_function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #37
#--------------------------

```
def test_xpath_contains_function_single_string_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})()]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result == xpath
```


# LLM-generated content at query #38
#--------------------------

```
def test_gt_function_with_non_number_argument_type():
    from pyquery.jquerytranslator import JQueryTranslator
    from cssselect.parser import Function
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [])
    function.argument_types = lambda: ['STRING']
    try:
        translator.xpath_gt_function(xpath, function)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)
        assert "Expected a single integer for :gt()" in str(e)
```


# LLM-generated content at query #39
#--------------------------

```
def test_xpath_eq_function_returns_correct_xpath_for_index_zero():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions[-1] == 'position() = 1'

def test_xpath_eq_function_returns_correct_xpath_for_positive_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions[-1] == 'position() = 3'

def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except Exception:
        assert True
```


# LLM-generated content at query #40
#--------------------------

```
def test_xpath_gt_function_raises_on_string_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from lxml.cssselect import FunctionElement, ExpressionError
    translator = JQueryTranslator()
    xpath = translator.xpath_for_class('div')
    function = FunctionElement('gt', ['"string"'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


