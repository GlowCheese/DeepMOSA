####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_xpath_eq_function_with_valid_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ['position() = 1']

def test_xpath_eq_function_with_invalid_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #2
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']

    xpath = XPathExpr()
    function = Function('lt', [Number(1)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 2']

    xpath = XPathExpr()
    function = Function('lt', [String('invalid')])
    try:
        translator.xpath_lt_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #3
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert isinstance(translator, JQueryTranslator)
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #4
#--------------------------

```python
def test_xpath_eq_function_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['test'])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ('test',)"


# LLM-generated content at query #5
#--------------------------

```python
def test_lower_case_attribute_names_false_when_xhtml_true():
    translator = JQueryTranslator(xhtml=True)
    assert translator.lower_case_attribute_names is False


# LLM-generated content at query #6
#--------------------------

```python
def test_lower_case_attribute_names_initialization():
    translator = JQueryTranslator(xhtml=False)
    assert translator.lower_case_attribute_names == True


# LLM-generated content at query #7
#--------------------------

```python
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['STRING'], arguments=[Mock(value='test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['IDENT'], arguments=[Mock(value='test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=[Mock(value=123)])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Mock(value=123)]"


# LLM-generated content at query #8
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function(None, [String(None, 'invalid')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_has_function_with_valid_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)

def test_xpath_has_function_with_invalid_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert "Expected a single string or ident for :has(), got" in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//div')
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=[Mock(value='0')])
    result = translator.xpath_gt_function(xpath, function)
    assert result.get_condition() == '//div[position() > 1]'


# LLM-generated content at query #11
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=['invalid'])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['invalid']"


# LLM-generated content at query #12
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='test')]
    translator.xpath_literal = Mock(return_value='"test"')
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
    xpath.add_post_condition.assert_called_once_with('contains(., "test")')


# LLM-generated content at query #13
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    function = type('MockFunction', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(None, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [MockArg(value=123)]"


# LLM-generated content at query #14
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//p')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': '0'})()], 'argument_types': lambda: ['NUMBER']})
    result = translator.xpath_gt_function(xpath, function)
    assert result.get_condition() == '//p[position() > 1]'


# LLM-generated content at query #15
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_arg():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #16
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('lt', [String('invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #17
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('contains', [Number(1)])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #18
#--------------------------

```python
def test_xpath_contains_function_with_valid_string():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=[Mock(value='test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = ['invalid']

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(Mock(), function)

    assert "Expected a single string or ident for :contains(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #21
#--------------------------

```python
def test_xpath_lt_function_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=['test'])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #22
#--------------------------

```python
def test_xpath_gt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #23
#--------------------------

```python
def test_xpath_gt_function_raises_expression_error():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    assert_raises(ExpressionError, translator.xpath_gt_function, xpath, function)


# LLM-generated content at query #24
#--------------------------

```python
def test_xpath_eq_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='test')]

    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['test']"


# LLM-generated content at query #25
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    assert_raises(ExpressionError, translator.xpath_eq_function, xpath, function)


# LLM-generated content at query #26
#--------------------------

```python
def test_xpath_has_function_predicate():
    class MockFunction:
        def __init__(self, args):
            self.arguments = args

        def argument_types(self):
            return [type(arg).__name__ for arg in self.arguments]

    function = MockFunction(['STRING'])
    assert function.argument_types() in (['STRING'], ['IDENT'])


# LLM-generated content at query #27
#--------------------------

```python
def test_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Number(0)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 1']

    xpath = XPathExpr()
    function = Function('eq', [Number(1)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 2']

    xpath = XPathExpr()
    function = Function('eq', [String('invalid')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ('invalid',)"


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_conditions == ['position() > 1']

    function = Function('gt', [Number(1)])
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_conditions == ['position() > 1', 'position() > 2']

    try:
        function = Function('gt', [String('invalid')])
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #29
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']

    xpath = XPathExpr()
    function = Function('lt', [Number(1)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 2']

    xpath = XPathExpr()
    function = Function('lt', [String('invalid')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #30
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '123'})()]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #31
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #32
#--------------------------

```python
def test_xpath_lt_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function(identifier='lt', arguments=[String(value='invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #33
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('.//h1')
    function = Function('contains', [String('title')])
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == './/h1[contains(., "title")]'


# LLM-generated content at query #34
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #35
#--------------------------

```python
def test_xpath_has_function_predicate():
    translator = JQueryTranslator()
    function = type('MockFunction', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('MockArg', (), {'value': 'test'})()]})()
    assert translator.xpath_has_function(None, function) is None


# LLM-generated content at query #36
#--------------------------

```python
def test_xpath_has_function_predicate_true():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=[Mock(value='test')])
    assert translator.xpath_has_function(xpath, function) == xpath


# LLM-generated content at query #37
#--------------------------

```python
def test_xpath_has_function_with_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)

def test_xpath_has_function_with_non_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]' in str(xpath)

def test_xpath_has_function_with_tag_selector():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert 'descendant::div' in str(xpath)

def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :has(), got" in str(e)


# LLM-generated content at query #38
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == 'position() > 1'


# LLM-generated content at query #39
#--------------------------

```python
def test_xpath_has_function_with_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == './/*[descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]]'

def test_xpath_has_function_with_non_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == './/*[descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]]'

def test_xpath_has_function_with_tag_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == './/*[descendant::div]'

def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #40
#--------------------------

```python
def test_xpath_contains_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = mock.Mock()
    function.argument_types.return_value = ['INVALID_TYPE']
    function.arguments = ['invalid_arg']

    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError was not raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got ['invalid_arg']"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('.//*')
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == './/*[position() > 1]'

    function = Function('gt', [Number(1)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == './/*[position() > 2]'


# LLM-generated content at query #2
#--------------------------

```python
def test_xpath_eq_function_valid_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ['position() = 1']

def test_xpath_eq_function_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ('invalid',)"


# LLM-generated content at query #3
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert isinstance(translator, JQueryTranslator)
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #4
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ["descendant::div"]


# LLM-generated content at query #5
#--------------------------

```python
def test_lower_case_attribute_names_when_xhtml_is_false():
    translator = JQueryTranslator(xhtml=False)
    assert translator.lower_case_attribute_names is True


# LLM-generated content at query #6
#--------------------------

```python
def test_xpath_lt_function_with_valid_index():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'arguments': [type('Argument', (), {'value': '0'})], 'argument_types': lambda: ['NUMBER']})
    result = translator.xpath_lt_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ['position() < 1']


# LLM-generated content at query #7
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #8
#--------------------------

```python
def test_xpath_gt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('gt', [String('invalid')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=['invalid'])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got 'invalid'"


# LLM-generated content at query #10
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_conditions == ['position() > 1']

    xpath = XPathExpr()
    function = Function('gt', [Number(1)])
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_conditions == ['position() > 2']

    xpath = XPathExpr()
    function = Function('gt', [String('invalid')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #11
#--------------------------

```python
def test_xpath_eq_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    assert translator.xpath_eq_function(xpath, function) == None


# LLM-generated content at query #12
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [String('invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #13
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('eq', [String('invalid')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #14
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']

    xpath = XPathExpr()
    function = Function('lt', [Number(1)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 2']

    xpath = XPathExpr()
    function = Function('lt', [String('invalid')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #15
#--------------------------

```python
def test_lower_case_attribute_names_default():
    translator = JQueryTranslator()
    assert translator.lower_case_attribute_names is True


# LLM-generated content at query #16
#--------------------------

```python
def test_xpath_has_function_with_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.bar'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]' in str(xpath)

def test_xpath_has_function_with_non_matching_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.baz'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]' in str(xpath)

def test_xpath_has_function_with_element_selector():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'div'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert 'descendant::div' in str(xpath)

def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 123})],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except Exception as e:
        assert "Expected a single string or ident for :has()" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_xpath_gt_function_with_non_number_argument():
    translator = JQueryTranslator()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = ['invalid']
    try:
        translator.xpath_gt_function(Mock(), function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #18
#--------------------------

```python
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})()]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['IDENT'], 'arguments': [type('Arg', (), {'value': 'test'})()]})()
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_invalid_arg():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': 123})()]})()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #19
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_contains_function_with_string_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=[MockArgument(value='test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath


# LLM-generated content at query #21
#--------------------------

```python
def test_xpath_gt_function_with_valid_integer():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=[Mock(value='0')])
    result = translator.xpath_gt_function(xpath, function)
    assert result is xpath
    assert xpath.conditions == ['position() > 1']


# LLM-generated content at query #22
#--------------------------

```python
def test_xpath_has_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})()]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #23
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('contains', [String('test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.get_xpath() == '//div[contains(., "test")]'

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('contains', [Ident('test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.get_xpath() == '//div[contains(., "test")]'

def test_xpath_contains_function_invalid_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('contains', [Number(123)])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(123)]"


# LLM-generated content at query #24
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']


# LLM-generated content at query #25
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']

    xpath = XPathExpr()
    function = Function('lt', [Number(1)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 2']


# LLM-generated content at query #26
#--------------------------

```python
def test_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Number(0)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 1']

    xpath = XPathExpr()
    function = Function('eq', [Number(1)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 2']

    xpath = XPathExpr()
    function = Function('eq', [String('invalid')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #27
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=['123'])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(Mock(), function)


# LLM-generated content at query #29
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_arguments():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=['123'])
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got '123'"


# LLM-generated content at query #30
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #31
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = ['invalid']
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #32
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #33
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('has', [String('".bar"')])
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == '//div[descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]]'


# LLM-generated content at query #34
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [Number(1)])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(1)]"


# LLM-generated content at query #35
#--------------------------

```python
def test_xpath_gt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #36
#--------------------------

```python
def test_xpath_gt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [String('0')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['0']"


# LLM-generated content at query #37
#--------------------------

```python
def test_xpath_contains_function_raises_expression_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock()
    function.argument_types.return_value = ['INVALID']
    function.arguments = ['invalid_arg']
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath, function)
    assert "Expected a single string or ident for :contains(), got ['invalid_arg']" in str(excinfo.value)


# LLM-generated content at query #38
#--------------------------

```python
def test_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('eq', [Number(0)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 1']

    xpath = XPathExpr()
    function = Function('eq', [Number(1)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 2']


# LLM-generated content at query #39
#--------------------------

```python
def test_xpath_gt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    try:
        translator.xpath_gt_function(Mock(), function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #40
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [String('test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_conditions == ["contains(., 'test')"]


