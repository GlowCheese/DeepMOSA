####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(value=123)]"


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_hidden_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test that the method adds the correct condition for hidden input elements
    translator.xpath_hidden_pseudo(xpath)
    assert xpath.condition == "@type = 'hidden' and name(.) = 'input'"

    # Test that the XPath expression is correctly formatted
    assert str(xpath) == "*[@type = 'hidden' and name(.) = 'input']"


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': 'invalid'})],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='0')]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 1]'
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='2')]
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 3]'
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = Mock()
    function.argument_types.return_value = ['STRING']
    function.arguments = [Mock(value='invalid')]
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_eq_function(xpath, function)
    assert "Expected a single integer for :eq(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with a simple selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with an ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == 'Expected a single string or ident for :has(), got [NUMBER number 123]'


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"

    # Test with zero index
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with ident argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})],
        'argument_types': lambda self: ['STRING']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})],
        'argument_types': lambda self: ['IDENT']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '123'})],
        'argument_types': lambda self: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[self::bar]'
    assert result.path == ''
    assert result.element == '*'
    assert result.condition == ''

    # Test with valid ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'
    assert result.path == ''
    assert result.element == '*'
    assert result.condition == ''

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"

    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING', 'STRING'],
        'arguments': [
            type('Argument', (), {'value': '.bar'})(),
            type('Argument', (), {'value': '.baz'})()
        ]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [.bar, .baz]"


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::div"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(value=123)]"


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'
    assert xpath.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'
    assert xpath.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with a valid selector
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with a valid element selector
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number('123')]"

    # Test with empty string
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': ''})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "")]'


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(value=123)]"


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert str(xpath) == '[*][position() = 1]'

    # Test with another valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert str(xpath) == '[*][position() = 3]'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #35
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result == xpath
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['IDENT']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('test')]"

    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [
            type('Argument', (), {'value': 'test1'})(),
            type('Argument', (), {'value': 'test2'})()
        ],
        'argument_types': lambda: ['STRING', 'STRING']
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got ['STRING', 'STRING']"


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)

    # Test with negative number
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'

    # Test with large number
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '999'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1000'


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt(), got" in str(e)

    # Test with zero index
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'


# LLM-generated content at query #41
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #42
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #43
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
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

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got (NUMBER 123,)"


# LLM-generated content at query #44
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid selector
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #45
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == '//*[contains(., "test")]'

    # Test with IDENT argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == '//*[contains(., "test")]'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #46
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt()" in str(e)

    # Test with zero index
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'


# LLM-generated content at query #47
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #48
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "*[contains(., 'test')]"

    # Test with ident argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "*[contains(., 'test')]"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER (123)]"


# LLM-generated content at query #49
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER (123)]"


# LLM-generated content at query #50
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['INVALID'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Argument(value='.bar')]"


# LLM-generated content at query #51
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #52
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #53
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('123')]"


# LLM-generated content at query #54
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with another valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt(), got" in str(e)


# LLM-generated content at query #55
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 1'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #56
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
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 1'

    # Test with another valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 3'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #57
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #58
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    xpath = XPathExpr()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    xpath = XPathExpr()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #59
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #60
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #61
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid index
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert str(xpath) == '*[position() = 1]'

    # Test with another valid index
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert str(xpath) == '*[position() = 3]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #62
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 1]'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 3]'

    # Test with invalid argument type
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


# LLM-generated content at query #63
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)


# LLM-generated content at query #64
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with a simple selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with an ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ('123',)"


# LLM-generated content at query #65
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got (123,)"


# LLM-generated content at query #66
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
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Argument(value='123')]"


# LLM-generated content at query #67
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #68
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 1]'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 3]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)


# LLM-generated content at query #69
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #70
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['INVALID'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Arg(value='.bar')]"


# LLM-generated content at query #71
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.bar'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.baz'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'div'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 123})],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #72
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
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

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #73
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #74
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda self: ['STRING']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == 'contains(., "test")'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda self: ['IDENT']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == 'contains(., "test")'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '123'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #75
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #76
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with a string argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with an ident argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['IDENT']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with an invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '123'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('123')]"


# LLM-generated content at query #77
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)

    # Test with zero index
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with negative index
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '-1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 0'


# LLM-generated content at query #78
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER (123)]"


# LLM-generated content at query #79
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with a string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with an ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == 'Expected a single string or ident for :contains(), got [123]'


# LLM-generated content at query #80
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['INVALID'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Argument(value='.bar')]"


# LLM-generated content at query #81
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.bar'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.baz'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'div'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 123})],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #82
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #83
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #84
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': '1'})],
        'argument_types': lambda: ['NUMBER']
    })
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Arg', (), {'value': 'invalid'})],
        'argument_types': lambda: ['STRING']
    })
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "ExpressionError not raised"
    except ExpressionError as e:
        assert "Expected a single integer for :gt()" in str(e)


# LLM-generated content at query #85
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)


# LLM-generated content at query #86
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert str(result) == '[*][position() < 2]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #87
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['INVALID'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #88
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 1]'

    # Test with another valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 3]'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #89
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 1'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)

    # Test with different number values
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 3'


# LLM-generated content at query #90
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #91
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"

    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING', 'STRING'],
        'arguments': [
            type('Argument', (), {'value': 'test1'})(),
            type('Argument', (), {'value': 'test2'})()
        ]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got ['test1', 'test2']"


# LLM-generated content at query #92
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #93
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "position() = 1"

    # Test with another valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "position() = 3"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_disabled_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test disabled input
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled button
    xpath = translator.xpathexpr_cls(element='button')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled select
    xpath = translator.xpathexpr_cls(element='select')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled option
    xpath = translator.xpathexpr_cls(element='option')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled optgroup
    xpath = translator.xpathexpr_cls(element='optgroup')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled textarea
    xpath = translator.xpathexpr_cls(element='textarea')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None

    # Test disabled fieldset
    xpath = translator.xpathexpr_cls(element='fieldset')
    result = translator.xpath_disabled_pseudo(xpath)
    assert result.condition == translator._format_disabled_xpath()
    assert result.post_condition is None


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('123')]"

    # Test with empty string
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': ''})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., '')"


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid selector
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ('123',)"


# LLM-generated content at query #4
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    translator.xpath_input_pseudo(xpath)
    assert xpath.condition == "(name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button')"


# LLM-generated content at query #6
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #8
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '0'})],
        'argument_types': lambda: ['NUMBER']
    })
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*' + '[position() = 1]'
    assert result.post_condition == 'position() = 1'

    # Test with another valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '2'})],
        'argument_types': lambda: ['NUMBER']
    })
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*' + '[position() = 3]'
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})],
        'argument_types': lambda: ['STRING']
    })
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #9
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 1'

    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #10
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #12
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('123')]"


# LLM-generated content at query #13
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type (should raise ExpressionError)
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


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [Mock(value='1')]

    result = translator.xpath_lt_function(xpath, function)
    assert result is xpath
    assert xpath.post_condition == 'position() < 2'

    # Test with invalid argument type
    function.argument_types.return_value = ['STRING']
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == '//*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    xpath = XPathExpr()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    xpath = XPathExpr()
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_contains_function(xpath, function)
    assert "Expected a single string or ident for :contains()" in str(excinfo.value)


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == 'position() = 1'

    # Test with another valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == 'position() = 3'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "*[contains(., 'test')]"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "*[contains(., 'test')]"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda self: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda self: ['STRING']
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number('123')]"


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with a simple selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with an ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with STRING argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #27
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 1]'

    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '[*][position() = 2]'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #29
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #31
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(123)]"


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER('123')]"


# LLM-generated content at query #33
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(value=123)]"


# LLM-generated content at query #34
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(value='123')]"


# LLM-generated content at query #35
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)


# LLM-generated content at query #36
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(value=123)]"


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.bar'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"
    assert result.post_condition == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'div'})],
        'argument_types': lambda: ['IDENT']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::div"
    assert result.post_condition == "descendant::div"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '123'})],
        'argument_types': lambda: ['NUMBER']
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()

    # Test with valid input
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid input (non-integer)
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)

    # Test with multiple arguments
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER', 'NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})(), type('Argument', (), {'value': '2'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #39
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"

    # Test with negative index
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #41
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with a simple selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with an ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::div"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #42
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"

    # Test with multiple arguments
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [
            type('Argument', (), {'value': '0'})(),
            type('Argument', (), {'value': '1'})()
        ]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['0', '1']"


# LLM-generated content at query #43
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with a string argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with an ident argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)


# LLM-generated content at query #44
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid integer argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #45
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "position() = 1"

    # Test with valid integer argument (non-zero)
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == "position() = 3"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['invalid']"


# LLM-generated content at query #46
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #47
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with invalid argument type
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

    # Test with negative number
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 0'


# LLM-generated content at query #48
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with another valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :gt(), got" in str(e)


# LLM-generated content at query #49
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with non-matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == "descendant::div"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #50
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"

    # Test with no arguments
    function = type('Function', (), {
        'arguments': [],
        'argument_types': lambda: []
    })()
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got []"


# LLM-generated content at query #51
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with string argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #52
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert str(result) == '[*][position() > 2]'
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


# LLM-generated content at query #53
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with string argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['IDENT']
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '123'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #54
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with string argument
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert str(result) == 'descendant-or-self::*[contains(., "test")]'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"


# LLM-generated content at query #55
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })
    translator.xpath_has_function(xpath, function())
    assert str(xpath) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with non-matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.baz'})()]
    })
    translator.xpath_has_function(xpath, function())
    assert str(xpath) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })
    translator.xpath_has_function(xpath, function())
    assert str(xpath) == "descendant::div"

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })
    try:
        translator.xpath_has_function(xpath, function())
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Argument(value='123')]"


# LLM-generated content at query #56
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #57
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '1'})()]
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #58
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type
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

    # Test with multiple arguments
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER', 'NUMBER'],
        'arguments': [
            type('Argument', (), {'value': '0'})(),
            type('Argument', (), {'value': '1'})()
        ]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :eq(), got ['0', '1']"


# LLM-generated content at query #59
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

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #60
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with another valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #61
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #62
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with selector matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with selector not matching descendant
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['INVALID'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #63
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

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_lt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


