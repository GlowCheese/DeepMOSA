####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    function = Function('eq', [String('0')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ('0',)"


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_constructor():
    translator = JQueryTranslator()
    assert isinstance(translator, JQueryTranslator)
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #3
#--------------------------

```python
def test_xpath_has_function_with_matching_selector():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(arguments=[MockArgument(value='.bar')], argument_types=lambda: ['STRING'])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]']

def test_xpath_has_function_with_non_matching_selector():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(arguments=[MockArgument(value='.baz')], argument_types=lambda: ['STRING'])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]']

def test_xpath_has_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(arguments=[MockArgument(value=123)], argument_types=lambda: ['NUMBER'])
    try:
        translator.xpath_has_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #4
#--------------------------

```python
def test_lower_case_names_when_not_xhtml():
    translator = JQueryTranslator(xhtml=False)
    assert translator.lower_case_element_names is True
    assert translator.lower_case_attribute_names is True


# LLM-generated content at query #5
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == 'position() > 1'

    function = Function('gt', [Number(1)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == 'position() > 2'

    function = Function('gt', [Number(2)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == 'position() > 3'


# LLM-generated content at query #6
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #7
#--------------------------

```python
def test_xpath_gt_function_raises_ExpressionError_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #8
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//div')
    function = type('Function', (), {'arguments': [type('Argument', (), {'value': '0'})()], 'argument_types': lambda: ['NUMBER']})
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']


# LLM-generated content at query #9
#--------------------------

```python
def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='invalid')]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)

    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #10
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {'arguments': [type('Argument', (), {'value': '.bar'})], 'argument_types': lambda: ['STRING']})
    result = translator.xpath_has_function(xpath, function)
    assert result is xpath
    assert xpath.post_conditions == ['descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]']


# LLM-generated content at query #11
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('gt', [Number(0)])
    translator.xpath_gt_function(xpath, function)
    assert str(xpath) == '//div[position() > 1]'


# LLM-generated content at query #12
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function("contains", [String("test")])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
    assert "contains(., 'test')" in xpath.post_conditions


# LLM-generated content at query #13
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('//div')
    function = Mock(argument_types=lambda: ['STRING'], arguments=[Mock(value='test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]


# LLM-generated content at query #14
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #15
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function(Function.PREFIX, 'has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']

    xpath = XPathExpr()
    function = Function(Function.PREFIX, 'has', [Ident('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']

    xpath = XPathExpr()
    function = Function(Function.PREFIX, 'has', [Number(1)])
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(1)]"


# LLM-generated content at query #16
#--------------------------

```python
def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #17
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #18
#--------------------------

```python
def test_xpath_gt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = MockFunction(argument_types=lambda: ['STRING'], arguments=['test'])
    assert raises(ExpressionError, translator.xpath_gt_function, xpath, function)


# LLM-generated content at query #19
#--------------------------

```python
def test_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=[Mock(value='0')])
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_conditions == ['position() > 1']


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['INVALID'], arguments=['test'])
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #21
#--------------------------

```python
def test_xpath_contains_function_valid_arguments():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': 'test'})()]})()
    assert translator.xpath_contains_function(xpath, function) == xpath


# LLM-generated content at query #22
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #23
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
    function = Function('lt', [Number(2)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 3']


# LLM-generated content at query #24
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('descendant-or-self::*')
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']


# LLM-generated content at query #25
#--------------------------

```python
def test_xpath_lt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('lt', [String('test')])
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['test']"


# LLM-generated content at query #26
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #27
#--------------------------

```python
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': 'test'})], 'argument_types': lambda: ['STRING']})
    result = translator.xpath_contains_function(xpath, function)
    assert result.get_condition() == ".//*[contains(., 'test')]"
    assert result.get_post_condition() == "contains(., 'test')"

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': 'test'})], 'argument_types': lambda: ['IDENT']})
    result = translator.xpath_contains_function(xpath, function)
    assert result.get_condition() == ".//*[contains(., 'test')]"
    assert result.get_post_condition() == "contains(., 'test')"

def test_xpath_contains_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('.//*')
    function = type('Function', (), {'arguments': [type('Arg', (), {'value': '123'})], 'argument_types': lambda: ['NUMBER']})
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': '1'})()]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ('1',)"


# LLM-generated content at query #29
#--------------------------

```python
def test_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('eq', [Number(0)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 1']

    xpath = XPathExpr('')
    function = Function('eq', [Number(1)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ['position() = 2']

    xpath = XPathExpr('')
    function = Function('eq', [String('0')])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['0']"


# LLM-generated content at query #30
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [String('test')])
    result = translator.xpath_contains_function(xpath, function)
    assert result is xpath
    assert 'contains(., "test")' in xpath.post_conditions


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_xpath_submit_pseudo():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    translator.xpath_submit_pseudo(xpath)
    assert xpath.condition == "@type = 'submit' and name(.) = 'input'"


# LLM-generated content at query #2
#--------------------------

```python
def test_xpath_disabled_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    translator.xpath_disabled_pseudo(xpath)
    assert str(xpath).endswith("((name(.) = 'button' or name(.) = 'input' or name(.) = 'select' or name(.) = 'textarea' or name(.) = 'fieldset') and @disabled or (ancestor::fieldset[@disabled] and not(ancestor::legend[not(preceding-sibling::legend)]))) or ((name(.) = 'option' and @disabled or ancestor::optgroup[@disabled])) or ((name(.) = 'optgroup' and @disabled))")


# LLM-generated content at query #3
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']


# LLM-generated content at query #4
#--------------------------

```python
def test_xpath_has_function_predicate():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda: ['STRING'], 'arguments': [type('Arg', (), {'value': '.bar'})()]})()
    translator.xpath_has_function(xpath, function)
    assert True


# LLM-generated content at query #5
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
    function = Function('gt', [String('0')])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('0',)"


# LLM-generated content at query #6
#--------------------------

```python
def test_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function("eq", [Number(0)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ["position() = 1"]

    xpath = XPathExpr()
    function = Function("eq", [Number(1)])
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_conditions == ["position() = 2"]

    xpath = XPathExpr()
    function = Function("eq", [String("0")])
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['0']"


# LLM-generated content at query #7
#--------------------------

```python
def test_jquerytranslator_constructor():
    translator = JQueryTranslator()
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #8
#--------------------------

```python
def test_jquery_translator_constructor():
    translator = JQueryTranslator()
    assert isinstance(translator, JQueryTranslator)
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #9
#--------------------------

```python
def test_jquerytranslator_constructor():
    translator = JQueryTranslator()
    assert isinstance(translator, JQueryTranslator)
    assert translator.xpathexpr_cls == XPathExpr


# LLM-generated content at query #10
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #11
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
    function = Function('lt', [Number(2)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 3']


# LLM-generated content at query #12
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [String('invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #13
#--------------------------

```python
def test_xpath_eq_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    assert_raises(ExpressionError, translator.xpath_eq_function, xpath, function)


# LLM-generated content at query #14
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('//div')
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert str(xpath) == '//div[position() < 1]'

    xpath = XPathExpr('//div')
    function = Function('lt', [Number(1)])
    translator.xpath_lt_function(xpath, function)
    assert str(xpath) == '//div[position() < 2]'

    xpath = XPathExpr('//div')
    function = Function('lt', [String('invalid')])
    try:
        translator.xpath_lt_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #15
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [String('test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [Ident('test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_invalid_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [Number(123)])
    try:
        translator.xpath_contains_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(123)]"


# LLM-generated content at query #16
#--------------------------

```python
def test_xpath_gt_function_raises_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #17
#--------------------------

```python
def test_xpath_contains_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function(Function.PSEUDO, 'contains', [Number(1)])
    assert translator.xpath_contains_function(xpath, function) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('contains', [Number(1)])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #19
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='invalid')]

    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #20
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('')
    function = type('Function', (), {'arguments': [type('Argument', (), {'value': '0'})()]})()
    result = translator.xpath_lt_function(xpath, function)
    assert result.get_condition() == 'position() < 1'


# LLM-generated content at query #21
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('lt', [Number(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_conditions == ['position() < 1']


# LLM-generated content at query #22
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('descendant-or-self::*')
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']


# LLM-generated content at query #23
#--------------------------

```python
def test_xpath_lt_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    mock_xpath = MagicMock()
    mock_function = MagicMock()
    mock_function.argument_types.return_value = ['STRING']
    mock_function.arguments = [MagicMock(value='invalid')]

    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(mock_xpath, mock_function)


# LLM-generated content at query #24
#--------------------------

```python
def test_xpath_has_function_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls('')

    class MockFunction:
        def __init__(self, arg_types):
            self.arg_types = arg_types
            self.arguments = [type('obj', (object,), {'value': 'test'})()]

        def argument_types(self):
            return self.arg_types

    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, MockFunction(['NUMBER']))


# LLM-generated content at query #25
#--------------------------

```python
def test_xpath_eq_function_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #26
#--------------------------

```python
def test_xpath_contains_function_with_string():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_ident():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]

def test_xpath_contains_function_with_invalid_argument_type():
    translator = JQueryTranslator()
    xpath = translator.xpath_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #27
#--------------------------

```python
def test_xpath_gt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = mock.Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = ['invalid']

    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #28
#--------------------------

```python
def test_xpath_contains_function_raises_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    function = type('MockFunction', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('MockArg', (), {'value': 123})()]})()
    try:
        translator.xpath_contains_function(None, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got (MockArg(value=123),)"


# LLM-generated content at query #29
#--------------------------

```python
def test_xpath_lt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='invalid')]

    try:
        translator.xpath_lt_function(xpath, function)
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #30
#--------------------------

```python
def test_xpath_gt_function_raises_error_when_argument_is_not_number():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #31
#--------------------------

```python
def test_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('contains', [String('test')])
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_conditions == ["contains(., 'test')"]


# LLM-generated content at query #32
#--------------------------

```python
def test_xpath_contains_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['NUMBER'], arguments=['123'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath, function)
    assert str(excinfo.value) == "Expected a single string or ident for :contains(), got ('123',)"


# LLM-generated content at query #33
#--------------------------

```python
def test_xpath_contains_function_raises_expression_error_for_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['INVALID'], arguments=['test'])
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #34
#--------------------------

```python
def test_xpath_gt_function_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Function('gt', [String('invalid')])
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #35
#--------------------------

```python
def test_xpath_eq_function_raises_expression_error_for_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = Mock(argument_types=lambda: ['STRING'], arguments=['invalid'])
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #36
#--------------------------

```python
def test_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Function('lt', [NumberToken(0)])
    translator.xpath_lt_function(xpath, function)
    assert xpath.path == 'position() < 1'


# LLM-generated content at query #37
#--------------------------

```python
def test_xpath_gt_function_with_non_number_argument():
    translator = JQueryTranslator()
    xpath = XPathExpr('')
    function = Mock(argument_types=lambda: ['STRING'], arguments=['text'])
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['text']"


# LLM-generated content at query #38
#--------------------------

```python
def test_xpath_has_function_with_invalid_argument_types():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {'argument_types': lambda: ['NUMBER'], 'arguments': [type('Arg', (), {'value': 123})()]})()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError to be raised"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER]"


# LLM-generated content at query #39
#--------------------------

```python
def test_xpath_has_function():
    translator = JQueryTranslator()
    xpath = XPathExpr('div')
    function = Function('has', [String('div')])
    translator.xpath_has_function(xpath, function)
    assert xpath.post_conditions == ['descendant::div']


# LLM-generated content at query #40
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


