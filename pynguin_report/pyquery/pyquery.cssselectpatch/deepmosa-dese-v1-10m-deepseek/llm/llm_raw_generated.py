####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_xpath_contains_function_returns_xpath_with_contains_condition():
    translator = JQueryTranslator()
    xpath = translator.xpath_expr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions[-1] == 'contains(., "text")'
```


# LLM-generated content at query #2
#--------------------------

```
def test_xpath_has_function_matches_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(
        XPathExpr('div', 'class="foo"'),
        Function('has', [Argument('STRING', '.bar')])
    )
    assert xpath.post_conditions == ["descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"]

def test_xpath_has_function_no_match():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(
        XPathExpr('div', 'class="foo"'),
        Function('has', [Argument('STRING', '.baz')])
    )
    assert xpath.post_conditions == ["descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"]

def test_xpath_has_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(
        XPathExpr('div'),
        Function('has', [Argument('IDENT', 'div')])
    )
    assert xpath.post_conditions == ["descendant::div"]

def test_xpath_has_function_raises_error_on_invalid_argument():
    translator = JQueryTranslator()
    from cssselect.xpath import ExpressionError
    try:
        translator.xpath_has_function(
            XPathExpr('div'),
            Function('has', [Argument('NUMBER', '1')])
        )
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #3
#--------------------------

```
def test_xpath_contains_function_passes_for_string_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.xpath import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [Token('STRING', '"title"')])
    translator.xpath_contains_function(xpath, function)

def test_xpath_contains_function_passes_for_ident_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.xpath import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [Token('IDENT', 'title')])
    translator.xpath_contains_function(xpath, function)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert translator is not None
    assert isinstance(translator, JQueryTranslator)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_xpath_has_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False
    except ExpressionError:
        assert True
```


# LLM-generated content at query #6
#--------------------------

```
def test_xpath_gt_function_positive_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']

def test_xpath_gt_function_negative_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '-1'})]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 0']

def test_xpath_gt_function_raises_error_for_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_gt_function_raises_error_for_multiple_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda: ['NUMBER', 'NUMBER'], 'arguments': [type('Arg', (), {'value': '0'}), type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #7
#--------------------------

```
def test_xpath_eq_function_single_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(arguments=[ArgumentMock(value='0')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_second_element():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(arguments=[ArgumentMock(value='1')])
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 2']


# LLM-generated content at query #8
#--------------------------

```
def test_xpath_eq_function_non_number_argument():
    mock_function = type('MockFunction', (object,), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (object,), {'value': 'not_a_number'})()]
    })()
    translator = JQueryTranslator()
    mock_xpath = type('MockXPath', (object,), {'add_post_condition': lambda self, cond: None})()
    try:
        translator.xpath_eq_function(mock_xpath, mock_function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'title'})()]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions[0] == "contains(., 'title')"

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'content'})()]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions[0] == "contains(., 'content')"

def test_xpath_contains_function_raises_on_invalid_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '42'})]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #10
#--------------------------

```python
def test_xpath_has_function_raises_error_for_non_string_or_ident_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [Number(1)])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #11
#--------------------------

```
def test_xpath_gt_function_accepts_number():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.cssselect_wrapper import XPathExpr, FunctionElement
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionElement(('gt',), [FunctionElement.NUMBER('0')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']
```


# LLM-generated content at query #12
#--------------------------

```
def test_xpath_lt_function_returns_condition_with_position_less_than_value_plus_one():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_with_zero_value():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_raises_error_on_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #13
#--------------------------

```
def test_xpath_contains_function_with_string_argument(self):
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='title')]
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_with_ident_argument(self):
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['IDENT']
    function.arguments = [MagicMock(value='title')]
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_raises_error_for_invalid_argument_type(self):
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value=1)]
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_contains_function_raises_error_for_multiple_arguments(self):
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['STRING', 'STRING']
    function.arguments = [MagicMock(value='a'), MagicMock(value='b')]
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #14
#--------------------------

```
def test_xpath_gt_function_with_valid_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//div')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']


# LLM-generated content at query #15
#--------------------------

```
def test_xpath_has_function_raises_for_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['NUMBER'])
    try:
        translator.xpath_has_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #16
#--------------------------

```
def test_xpath_eq_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': '2'})], 'argument_types': lambda self: ['NUMBER']})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3']

def test_xpath_eq_function_with_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': '0'})], 'argument_types': lambda self: ['NUMBER']})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_negative_index_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': '-1'})], 'argument_types': lambda self: ['NUMBER']})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except ValueError:
        assert True

def test_xpath_eq_function_with_non_number_argument_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': 'abc'})], 'argument_types': lambda self: ['STRING']})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except Exception:
        assert True
```


# LLM-generated content at query #17
#--------------------------

```
def test_xpath_eq_function_with_valid_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = lambda: None
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('arg', (), {'value': '0'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_second_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = lambda: None
    function.argument_types = lambda: ['NUMBER']
    function.arguments = [type('arg', (), {'value': '1'})()]
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 2']

def test_xpath_eq_function_with_non_number_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = lambda: None
    function.argument_types = lambda: ['STRING']
    function.arguments = [type('arg', (), {'value': 'test'})()]
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #18
#--------------------------

```
def test_xpath_gt_function_raises_error_for_non_number_argument():
    from pyquery.translator import JQueryTranslator, ExpressionError
    translator = JQueryTranslator()
    from cssselect.xpath import XPathExpr
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "ExpressionError not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #19
#--------------------------

```
def test_xpath_lt_function_with_non_number_arguments():
    from pyquery.translator import JQueryTranslator, XPathExpr, ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'foo'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```
def test_xpath_contains_function_accepts_string_argument_type():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.xpath import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [Token('STRING', 'test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #21
#--------------------------

def test_xpath_eq_function_predicate_true():
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import XPathExpr
    from pyquery.translator import ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '0'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result is not None


# LLM-generated content at query #22
#--------------------------

```
def test_xpath_lt_function_returns_xpath_with_position_less_than_one():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath_expr import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ["position() < 1"]

def test_xpath_lt_function_raises_error_for_non_number():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath_expr import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': ['test']})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except Exception as e:
        assert "Expected a single integer for :gt()" in str(e)
```


# LLM-generated content at query #23
#--------------------------

```
def test_xpath_gt_function_non_number_argument_types_raises_expression_error():
    from pyquery.jquery_translator import JQueryTranslator
    from cssselect.parser import Function
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [])
    function.argument_types = lambda: ['STRING']
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #24
#--------------------------

```
def test_xpath_eq_function_raises_error_for_non_number_argument():
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import ExpressionError
    from cssselect.parser import Function
    from cssselect.parser import parse
    xpath = JQueryTranslator().xpath_eq_function
    function = Function('eq', [parse('"string"')[0]])
    raised = False
    try:
        xpath(type('xpath', (), {'add_post_condition': lambda self, x: None})(), function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #25
#--------------------------

```
def test_xpath_eq_function_returns_correct_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_index_1():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3']

def test_xpath_eq_function_raises_error_for_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    raised = False
    try:
        translator.xpath_eq_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #26
#--------------------------

```
def test_xpath_eq_function_with_non_number_argument():
    function = type('MockFunction', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('MockArgument', (), {'value': 'abc'})()]})()
    xpath = type('MockXPath', (), {'add_post_condition': lambda self, cond: None})()
    translator = JQueryTranslator()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        assert True
```


# LLM-generated content at query #27
#--------------------------

def test_xpath_has_function_non_string_non_ident_raises_expression_error():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [NumericLiteral(42)])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_has_function_returns_elements_with_matching_descendant():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_tag('div'), MockFunction(['.bar']))
    assert xpath.path == 'descendant-or-self::div'
    assert xpath.post_conditions[0] == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

def test_xpath_has_function_returns_empty_when_no_match():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_tag('div'), MockFunction(['.baz']))
    assert xpath.path == 'descendant-or-self::div'
    assert xpath.post_conditions[0] == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

def test_xpath_has_function_works_with_tag_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(translator.xpath_for_tag('div'), MockFunction(['div']))
    assert xpath.path == 'descendant-or-self::div'
    assert xpath.post_conditions[0] == 'descendant::div'

def test_xpath_has_function_raises_error_on_invalid_argument_type():
    translator = JQueryTranslator()
    try:
        translator.xpath_has_function(translator.xpath_for_tag('div'), MockFunction([123]))
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #29
#--------------------------

```
def test_xpath_has_function_with_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//div')
    translator.xpath_has_function(xpath, function=lambda: None)
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    xpath = translator.xpath_cls(path='//div')
    result = translator.xpath_has_function(xpath, function)
    assert result is not None
```


# LLM-generated content at query #30
#--------------------------

```
def test_xpath_lt_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [Number('2')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_with_negative_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [Number('-1')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 0']

def test_xpath_lt_function_with_zero():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [Number('0')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_raises_error_on_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [String('abc')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass

def test_xpath_lt_function_raises_error_on_multiple_arguments():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = Function('lt', [Number('1'), Number('2')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Should have raised ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #31
#--------------------------

```python
def test_xpath_contains_function_raises_expression_error_on_invalid_argument_types(self):
    from pyquery.translator import JQueryTranslator, ExpressionError
    from cssselect.parser import Function
    from unittest.mock import Mock
    translator = JQueryTranslator()
    xpath = Mock()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='0')]
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #32
#--------------------------

```
def test_xpath_lt_function_valid_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.cssselectwrapper import XPathExpr
    from cssselect.parser import Function, Number
    translator = JQueryTranslator()
    xpath = XPathExpr('test')
    function = Function('lt', [Number('1')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 2']
```


# LLM-generated content at query #33
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value="title")]
    result = translator.xpath_contains_function(xpath, function)
    assert result.condition == "contains(., 'title')"

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['IDENT']
    function.arguments = [MagicMock(value="title")]
    result = translator.xpath_contains_function(xpath, function)
    assert result.condition == "contains(., 'title')"

def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value=1)]
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #34
#--------------------------

def test_xpath_gt_function_invalid_argument_type():
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('Arg', (), {'value': 'abc'})()]
    translator = JQueryTranslator()
    xpath = XPathExpr('div')
    try:
        translator.xpath_gt_function(xpath, MockFunction())
        assert False, "ExpressionError should have been raised"
    except ExpressionError:
        pass


# LLM-generated content at query #35
#--------------------------

```
def test_xpath_gt_function_raises_error_on_non_number():
    from pyquery.translator import JQueryTranslator
    from cssselect.parser import Function
    translator = JQueryTranslator()
    xpath = translator.xpath_gt_function(xpath, function)
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #36
#--------------------------

```
def test_xpath_gt_function_raises_expression_error_for_non_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.jquerytranslator import ExpressionError
    from pyquery.jquerytranslator import XPathExpr
    from lxml.cssselect import FunctionalPseudoElement
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionalPseudoElement('gt', ['string'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #37
#--------------------------

def test_xpath_has_function_accepts_string():
    from pyquery.jquery_translator import JQueryTranslator
    from cssselect.parser import Function, parse
    translator = JQueryTranslator()
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls('div'),
        Function('has', [parse('"test"')[0].parsed_tree])
    )


# LLM-generated content at query #38
#--------------------------

def test_xpath_lt_function_raises_error_on_non_number():
    from pyquery.jquery_translator import JQueryTranslator, ExpressionError
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = translator.xpath_for_pseudo('lt')
    function = Function('lt', [Token('STRING', 'foo')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #39
#--------------------------

```
def test_xpath_lt_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Argument', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_with_non_number_argument_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Argument', (), {'value': 'test'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #40
#--------------------------

```python
def test_xpath_contains_function_invalid_argument_type():
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock()]
    function.arguments[0].value = '123'
    translator = JQueryTranslator()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']


# LLM-generated content at query #2
#--------------------------

```
def test_xpath_gt_function_with_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.cssselectwrapper import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #3
#--------------------------

```
def test_xpath_lt_function_returns_correct_post_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})()]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_with_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_raises_error_for_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})()]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #4
#--------------------------

```python
def test_xpath_disabled_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//input')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.post_conditions == ['position() = 1']
    assert result.condition == '(@disabled or (ancestor::fieldset[@disabled and not(ancestor::legend[not(preceding-sibling::legend)])])) or (@disabled or ancestor::optgroup[@disabled]) or (@disabled)' 
```


# LLM-generated content at query #5
#--------------------------

```python
def test_xpath_has_function_matches_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/div')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

def test_xpath_has_function_no_match():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/div')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': '.baz'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

def test_xpath_has_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/div')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'div'})]})()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

def test_xpath_has_function_raises_on_invalid_args():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/div')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #6
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert translator is not None
    assert isinstance(translator.xpathexpr_cls, type)
```


# LLM-generated content at query #7
#--------------------------

```
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_conditions == ['position() > 1']
```


# LLM-generated content at query #8
#--------------------------

```python
from tests.utils import create_function_mock
xpath = object()
function = create_function_mock(argument_types=['NUMBER'])
result = xpath_gt_function(xpath, function)
assert function.argument_types() == ['NUMBER']
```


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_contains_function_returns_xpath_with_contains_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = DummyFunction(['STRING'], [DummyArgument('title')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = DummyFunction(['IDENT'], [DummyArgument('title')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_raises_expression_error_for_non_string_or_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = DummyFunction(['NUMBER'], [DummyArgument(1)])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

class DummyFunction:
    def __init__(self, argument_types, arguments):
        self._argument_types = argument_types
        self.arguments = arguments
    def argument_types(self):
        return self._argument_types

class DummyArgument:
    def __init__(self, value):
        self.value = value

class ExpressionError(Exception):
    pass
```


# LLM-generated content at query #10
#--------------------------

```
def test_xpath_eq_function_returns_first_element():
    xpath = XPathExpr()
    function = Function("eq", [Number("0")])
    result = JQueryTranslator().xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_returns_second_element():
    xpath = XPathExpr()
    function = Function("eq", [Number("1")])
    result = JQueryTranslator().xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 2']

def test_xpath_eq_function_raises_error_for_non_number():
    from cssselect.xpath import ExpressionError
    try:
        xpath = XPathExpr()
        function = Function("eq", [String("text")])
        JQueryTranslator().xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        pass

def test_xpath_eq_function_raises_error_for_multiple_arguments():
    from cssselect.xpath import ExpressionError
    try:
        xpath = XPathExpr()
        function = Function("eq", [Number("0"), Number("1")])
        JQueryTranslator().xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #11
#--------------------------

```
def test_xpath_has_function_invalid_argument_type_raises_error():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.exceptions import ExpressionError
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #12
#--------------------------

```
def test_xpath_lt_function_with_valid_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='2')]
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_with_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='0')]
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']

def test_xpath_lt_function_with_negative_number_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [MagicMock(value='-1')]
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_lt_function_with_invalid_argument_type_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #13
#--------------------------

```
def test_xpath_contains_function_with_string_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(['STRING'], [MockArgument('title')])
    result = translator.xpath_contains_function(xpath, function)
    assert function.argument_types() in (['STRING'], ['IDENT'])


# LLM-generated content at query #14
#--------------------------

```
def test_xpath_has_function_argument_types_is_STRING():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.xpath_expr import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    token = Token('string', '"test"', (0, 0))
    function = Function('has', [token], 'pseudo-class')
    translator.xpath_has_function(xpath, function)
```


# LLM-generated content at query #15
#--------------------------

```
def test_xpath_eq_function_raises_on_non_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('/test')
    function = type('MockFunction', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('MockArg', (), {'value': 'foo'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #16
#--------------------------

```
def test_xpath_gt_function_with_number_argument():
    from pyquery.pyquery import PyQuery
    from pyquery.cssselect import JQueryTranslator
    from cssselect.parser import parse
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    selectors = parse("h1:gt(0)")
    xpath = translator.selector_to_xpath(selectors[0])
    assert "position() > 1" in xpath
```


# LLM-generated content at query #17
#--------------------------

```python
def test_xpath_has_function_valid_argument_type_string():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MagicMock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [MagicMock(value='.bar')]
    translator.css_to_xpath = MagicMock(return_value='descendant::*[contains(concat(" ", @class, " "), " bar ")]')
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    translator.css_to_xpath.assert_called_once_with('.bar', prefix='descendant::')
```


# LLM-generated content at query #18
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls('test'), type('function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('arg', (object,), {'value': 'title'})]})())

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_contains_function(translator.xpathexpr_cls('test'), type('function', (object,), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('arg', (object,), {'value': 'title'})]})())

def test_xpath_contains_function_raises_expression_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    try:
        translator.xpath_contains_function(translator.xpathexpr_cls('test'), type('function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('arg', (object,), {'value': '1'})]})())
        assert False
    except ExpressionError:
        pass
```


# LLM-generated content at query #19
#--------------------------

```python
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = MockFunction(['STRING'], [MockArgument('"title"')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., '\"title\"')"]

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = MockFunction(['IDENT'], [MockArgument('title')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = MockFunction(['NUMBER'], [MockArgument('42')])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_contains_function_returns_updated_xpath():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/html/body')
    function = MockFunction(['STRING'], [MockArgument('"text"')])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_has_function_raises_expression_error_for_non_string_non_ident(self):
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import XPathExpr
    from pyquery.translator import ExpressionError
    from cssselect.parser import Function
    from cssselect.parser import parse
    translator = JQueryTranslator()
    xpath = XPathExpr()
    try:
        func = Function('has', [])
        func.argument_types = lambda: ['NUMBER']
        translator.xpath_has_function(xpath, func)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #21
#--------------------------

```
def test_xpath_lt_function_returns_xpath_with_correct_position_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result._post_conditions == ['position() < 2']

def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': 'text'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert 'Expected a single integer for :gt()' in str(e)
```


# LLM-generated content at query #22
#--------------------------

```
def test_xpath_contains_function_valid_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['STRING'], ['"test"'])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath

def test_xpath_contains_function_valid_ident():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = FunctionMock(['IDENT'], ['test'])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #23
#--------------------------

def test_expression_error_raised_for_non_number_argument():
    from pyquery.translator import JQueryTranslator
    translator = JQueryTranslator()
    xpath = object()
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'text'})]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False
    except ExpressionError:
        assert True


# LLM-generated content at query #24
#--------------------------

def test_xpath_gt_function_predicate_false():
    from pyquery.jquery_translator import JQueryTranslator
    from pyquery.cssselect import xpath as cssselect_xpath
    from lxml.cssselect import xpath as lxml_cssselect_xpath
    from pyquery.jquery_translator import XPathExpr
    from pyquery.jquery_translator import ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': 'test'})()]})()
    raised = False
    try:
        translator.xpath_gt_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised


# LLM-generated content at query #25
#--------------------------

def test_xpath_lt_function_raises_error_for_non_number():
    from pyquery.jquerytranslator import JQueryTranslator, ExpressionError
    from pyquery.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    try:
        translator.xpath_lt_function(xpath, MockFunction())
        assert False, "Expected ExpressionError"
    except ExpressionError:
        assert True


# LLM-generated content at query #26
#--------------------------

```
def test_lt_function_with_non_number_raises_expression_error():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function("lt", [Token("STRING", "text")])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #27
#--------------------------

```
def test_xpath_lt_function_with_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_lt_function_with_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']
```


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_gt_function_valid_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_gt_function(
        translator.xpathexpr_cls('test'),
        MockFunction(['NUMBER'], [MockArgument('0')]))
    assert xpath.post_conditions == ['position() > 1']
```


# LLM-generated content at query #29
#--------------------------

```
def test_xpath_eq_function_raises_error_on_non_number_argument():
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import ExpressionError
    from pyquery.translator import XPathExpr
    from pyquery.translator import cssselect_xpath
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('arg', (), {'value': 'test'})()]
    translator = JQueryTranslator()
    xpath = XPathExpr()
    try:
        translator.xpath_eq_function(xpath, MockFunction())
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #30
#--------------------------

```
def test_xpath_lt_function_with_non_number_argument_types():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.translator import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('FakeFunction', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': 'test'})]} )()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False
    except ExpressionError:
        assert True
```


# LLM-generated content at query #31
#--------------------------

def test_xpath_eq_function_valid_number():
    from pyquery.translator import JQueryTranslator, XPathExpr, ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_xpath_has_function_raises_error_for_non_string_non_ident_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '123'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


# LLM-generated content at query #33
#--------------------------

```
def test_xpath_lt_function_returns_xpath_with_position_condition():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '2'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 3']

def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'foo'})]})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except:
        pass

def test_xpath_lt_function_raises_error_for_empty_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: [], 'arguments': []})()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except:
        pass

def test_xpath_lt_function_with_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']
```


# LLM-generated content at query #34
#--------------------------

```
def test_xpath_contains_function_valid_string_argument():
    translator = JQueryTranslator()
    mock_xpath = XPathExpr()
    mock_function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArgument', (), {'value': 'test'})()],
        'value': 'test'
    })()
    result = translator.xpath_contains_function(mock_xpath, mock_function)
    assert result == mock_xpath
```


# LLM-generated content at query #35
#--------------------------

def test_xpath_gt_function_with_non_number_argument():
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import XPathExpr
    from pyquery.translator import ExpressionError
    from pyquery.translator import Function
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [lambda: type('Argument', (), {'value': 'not_a_number', 'argument_types': lambda: ['STRING']})()], ['STRING'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False
    except ExpressionError:
        pass


# LLM-generated content at query #36
#--------------------------

```
def test_xpath_eq_function_raises_error_for_non_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.translator import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Token('STRING', 'foo')])
    raised = False
    try:
        translator.xpath_eq_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #37
#--------------------------

```
def test_xpath_has_function_raises_expression_error_on_invalid_argument():
    from pyquery.translator import JQueryTranslator
    from cssselect.xpath import XPathExpr
    from cssselect.parser import Function, Token
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [Token('NUMBER', '42')])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #38
#--------------------------

```python
def test_xpath_has_function_raises_error_for_non_string_or_ident_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '123'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #39
#--------------------------

```
def test_xpath_contains_function_raises_for_invalid_argument():
    translator = JQueryTranslator()
    from lxml.cssselect import FunctionElement, ExpressionError
    from lxml.cssselect import xpath as xpath_mod
    xpath = xpath_mod.XPathExpr()
    function = FunctionElement('contains', [])
    function.argument_types = lambda: ['NUMBER']
    function.arguments = []
    raised = False
    try:
        translator.xpath_contains_function(xpath, function)
    except ExpressionError:
        raised = True
    assert raised
```


# LLM-generated content at query #40
#--------------------------

```
def test_xpath_contains_function_string():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/test')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'title'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_ident():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/test')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'content'})]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'content')"]

def test_xpath_contains_function_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls('/test')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False
    except Exception:
        assert True
```


# LLM-generated content at query #41
#--------------------------

```
def test_xpath_eq_function_with_valid_number():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '0'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_second_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '2'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['position() = 3']

def test_xpath_eq_function_with_non_number_raises_error():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    function = type('Function', (object,), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (object,), {'value': 'test'})()]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert 'Expected a single integer for :eq()' in str(e)

def test_xpath_eq_function_preserves_existing_conditions():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('test')
    xpath.add_post_condition('existing_condition')
    function = type('Function', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (object,), {'value': '1'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_conditions == ['existing_condition', 'position() = 2']
```


# LLM-generated content at query #42
#--------------------------

```
def test_xpath_eq_function_raises_expression_error_on_non_number_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.jquerytranslator import ExpressionError
    from pyquery.jquerytranslator import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = lambda: None
    function.arguments = [type('obj', (object,), {'value': 'text'})()]
    function.argument_types = lambda: ['STRING']
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError:
        pass
```


# LLM-generated content at query #43
#--------------------------

```
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'title'})]} )()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_with_ident_argument():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'title'})]} )()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'title')"]

def test_xpath_contains_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '5'})]} )()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass

def test_xpath_contains_function_returns_xpath():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})]} )()
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
```


# LLM-generated content at query #44
#--------------------------

```
def test_xpath_lt_function_inside_xpath_translator():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 2']
    assert result == xpath

def test_xpath_lt_function_zero_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 1']
    assert result == xpath

def test_xpath_lt_function_large_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '10'})]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_conditions == ['position() < 11']
    assert result == xpath
```


# LLM-generated content at query #45
#--------------------------

```python
def test_xpath_has_function_returns_elements_with_matching_descendant():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".bar")')
    assert len(result) == 1
    assert result[0].attrib['class'] == 'foo'

def test_xpath_has_function_returns_empty_when_no_matching_descendant():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".baz")')
    assert len(result) == 0

def test_xpath_has_function_returns_empty_when_self_matches_selector():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(".foo")')
    assert len(result) == 0

def test_xpath_has_function_works_with_element_selector():
    from pyquery.pyquery import PyQuery
    d = PyQuery('<div class="foo"><div class="bar"></div></div>')
    result = d('.foo:has(div)')
    assert len(result) == 1
    assert result[0].attrib['class'] == 'foo'
```


# LLM-generated content at query #46
#--------------------------

def test_xpath_has_function_raises_expression_error_for_non_string_non_ident_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False
    except ExpressionError:
        assert True


# LLM-generated content at query #47
#--------------------------

```
def test_xpath_eq_function_with_number_argument():
    from pyquery.translator import JQueryTranslator
    from pyquery.translator import XPathExpr
    from pyquery.translator import ExpressionError
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('obj', (object,), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('obj', (object,), {'value': '0'})]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result == xpath
```


# LLM-generated content at query #48
#--------------------------

```
def test_xpath_lt_function_with_non_number_argument_raises_expression_error():
    from pyquery.jquery_translator import JQueryTranslator
    from cssselect.parser import Function
    from cssselect.xpath import XPathExpr
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [])
    function.argument_types = lambda: ['STRING']
    function.arguments = [type('Arg', (), {'value': 'test'})()]
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert isinstance(e, ExpressionError)
        assert "Expected a single integer for :gt(), got" in str(e)
```


# LLM-generated content at query #49
#--------------------------

```
def test_xpath_contains_function_with_valid_string_argument():
    from pyquery.jquerytranslator import JQueryTranslator
    from pyquery.expression import ExpressionError
    translator = JQueryTranslator()
    class MockFunction:
        def argument_types(self):
            return ['STRING']
        arguments = [type('MockArgument', (), {'value': 'test'})()]
    class MockXPath:
        def add_post_condition(self, condition):
            pass
    xpath = MockXPath()
    result = translator.xpath_contains_function(xpath, MockFunction())
    assert result is not None
```


# LLM-generated content at query #50
#--------------------------

```
def test_xpath_eq_function_returns_correct_xpath_for_first_element():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '0'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.path == '//h1'
    assert result.post_conditions == ['position() = 1']

def test_xpath_eq_function_returns_correct_xpath_for_second_element():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})()]})()
    result = translator.xpath_eq_function(xpath, function)
    assert result.path == '//h1'
    assert result.post_conditions == ['position() = 2']

def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls(path='//h1')
    function = type('Function', (), {'argument_types': lambda self: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})()]})()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass
```


