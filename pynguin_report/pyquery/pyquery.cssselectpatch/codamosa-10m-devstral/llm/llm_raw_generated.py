####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with an ident selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #3
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

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
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [123]"

    # Test with special characters in string
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': "test's \"quotes\""})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test\\'s \"quotes\"')"


# LLM-generated content at query #4
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
    assert str(result) == 'position() = 1'

    # Test with another valid integer argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == 'position() = 3'

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


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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
        assert "Expected a single integer for :gt()" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::*[contains(concat(" ", @class, " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got [Number(value='123')]"


# LLM-generated content at query #8
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
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #9
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

    # Test with invalid selector type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == 'Expected a single string or ident for :has(), got [NUMBER("123")]'

    # Test with IDENT argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert str(result) == 'descendant::div'


# LLM-generated content at query #10
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
    assert result.post_condition == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

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
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)


# LLM-generated content at query #11
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching descendant
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('MockFunction', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('MockArg', (), {'value': '.bar'})()]
        })()
    )
    assert str(xpath) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with non-matching descendant
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('MockFunction', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('MockArg', (), {'value': '.baz'})()]
        })()
    )
    assert str(xpath) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' baz ')]"

    # Test with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('MockFunction', (), {
            'argument_types': lambda: ['STRING'],
            'arguments': [type('MockArg', (), {'value': 'div'})()]
        })()
    )
    assert str(xpath) == "descendant::div"

    # Test with invalid argument type
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('MockFunction', (), {
                'argument_types': lambda: ['INVALID'],
                'arguments': [type('MockArg', (), {'value': '.bar'})()]
            })()
        )


# LLM-generated content at query #12
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
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    xpath = XPathExpr()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    xpath = XPathExpr()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #13
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
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with IDENT argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    translator.xpath_contains_function(xpath, function)
    assert str(xpath) == "descendant-or-self::*[contains(., 'test')]"

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': 123})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got (123,)"


# LLM-generated content at query #14
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
    assert result.post_condition == "contains(., test)"

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
        assert str(e) == "Expected a single string or ident for :contains(), got (123,)"


# LLM-generated content at query #15
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert xpath.post_condition == 'position() > 2'
    assert result == xpath

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #16
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #17
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
    assert str(result) == '//*[position() < 2]'

    # Test with invalid argument type
    function_invalid = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})()],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function_invalid)


# LLM-generated content at query #18
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
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

    # Test with ident argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == "descendant::div"

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
        assert str(e) == "Expected a single string or ident for :has(), got (123,)"


# LLM-generated content at query #19
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
        assert str(e) == "Expected a single string or ident for :has(), got [123]"


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
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
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #21
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test basic functionality
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with IDENT argument type
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
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

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
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single string or ident for :contains()" in str(e)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})]
    })()
    translator.xpath_eq_function(xpath, function)
    assert xpath.post_condition == 'position() = 1'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_eq_function(xpath, function)


# LLM-generated content at query #26
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid number argument
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
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #27
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
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #28
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
    assert result is xpath
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #29
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
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #30
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid integer argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_JQueryTranslator_xpath_submit_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with input type submit
    result = translator.xpath_submit_pseudo(xpath)
    assert result.condition == "@type = 'submit' and name(.) = 'input'"

    # Test that the condition is properly added
    assert str(result) == "//*[@type = 'submit' and name(.) = 'input']"

    # Test with post_condition
    xpath.add_post_condition("position() = 1")
    result = translator.xpath_submit_pseudo(xpath)
    assert result.condition == "@type = 'submit' and name(.) = 'input'"
    assert result.post_condition == "position() = 1"
    assert str(result) == "//*[@type = 'submit' and name(.) = 'input'][position() = 1]"


# LLM-generated content at query #2
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()
    result = translator.xpath_input_pseudo(xpath)
    assert str(result) == "(name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button')"


# LLM-generated content at query #3
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
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #4
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
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #5
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = mock.Mock()
    function.argument_types.return_value = ['NUMBER']
    function.arguments = [mock.Mock(value='1')]

    result = translator.xpath_gt_function(xpath, function)
    assert result is xpath
    assert xpath.post_condition == 'position() > 2'

    # Test with invalid argument type
    function.argument_types.return_value = ['STRING']
    function.arguments = [mock.Mock(value='invalid')]

    with pytest.raises(ExpressionError) as excinfo:
        translator.xpath_gt_function(xpath, function)
    assert "Expected a single integer for :gt(), got ['invalid']" in str(excinfo.value)


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
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 2'

    # Test with zero index
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    xpath = XPathExpr()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 1'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    xpath = XPathExpr()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #7
#--------------------------

```python
def test_JQueryTranslator_xpath_input_pseudo():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test that input elements are matched
    result = translator.xpath_input_pseudo(xpath)
    assert str(result) == "descendant-or-self::*[((name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button'))]"

    # Test that the condition is added correctly
    assert xpath.condition == "(name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button')"


# LLM-generated content at query #8
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
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with ident argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['IDENT']
    })()
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #9
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

    # Test with IDENT argument type
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
        assert str(e) == "Expected a single string or ident for :has(), got [NUMBER('123')]"


# LLM-generated content at query #10
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
    assert result.post_condition == 'contains(., "test")'

    # Test with ident argument
    function = type('Function', (), {
        'argument_types': lambda self: ['IDENT'],
        'arguments': [type('Argument', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == 'contains(., "test")'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got ['NUMBER']"


# LLM-generated content at query #11
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
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': 123})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #12
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
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #13
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
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #14
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()

    # Test with STRING argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'test'})()]
    })()
    result = translator.xpath_contains_function(xpath, function)
    assert result.post_condition == "contains(., 'test')"

    # Test with IDENT argument
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
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #15
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
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #16
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
        assert "Expected a single integer for :gt()" in str(e)

    # Test with negative number
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '-1'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 0'

    # Test with zero
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_gt_function(xpath, function)
    assert result.post_condition == 'position() > 1'


# LLM-generated content at query #17
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '1'})],
        'argument_types': lambda: ['NUMBER']
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'invalid'})],
        'argument_types': lambda: ['STRING']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #18
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with valid selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.bar'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

    # Test with invalid argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"

    # Test with IDENT argument type
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda: ['IDENT'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'


# LLM-generated content at query #19
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

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
        'arguments': [type('Argument', (), {'value': 'test'})()],
        'argument_types': lambda: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #20
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 1'

    # Test with another valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '2'})()]
    })()
    translator.xpath_lt_function(xpath, function)
    assert xpath.post_condition == 'position() < 3'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_lt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ('invalid',)"


# LLM-generated content at query #21
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
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type (should raise ExpressionError)
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


# LLM-generated content at query #22
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #23
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
    try:
        translator.xpath_gt_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single integer for :gt(), got ['invalid']"


# LLM-generated content at query #24
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

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
    translator.xpath_contains_function(xpath, function)
    assert xpath.post_condition == "contains(., 'test')"

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '123'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_contains_function(xpath, function)


# LLM-generated content at query #25
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with invalid argument type
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '0'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


# LLM-generated content at query #26
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
        assert "Expected a single integer for :gt()" in str(e)


# LLM-generated content at query #27
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
        assert str(e) == "Expected a single string or ident for :contains(), got [Number(value=123)]"


# LLM-generated content at query #28
#--------------------------

```python
def test_JQueryTranslator_xpath_contains_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

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


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': '.baz'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::*[contains(concat(" ", @class, " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Arg', (), {'value': 'div'})()]
    })()
    translator.xpath_has_function(xpath, function)
    assert str(xpath) == 'descendant::div'

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
        assert str(e) == 'Expected a single string or ident for :has(), got [123]'


# LLM-generated content at query #31
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
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_has_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :has(), got ['NUMBER']"


# LLM-generated content at query #32
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()

    # Test with valid number argument
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '1'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 2'

    # Test with invalid argument type
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('Argument', (), {'value': 'invalid'})()]
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)

    # Test with zero index
    xpath = XPathExpr()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Argument', (), {'value': '0'})()]
    })()
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'


# LLM-generated content at query #33
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


# LLM-generated content at query #34
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

    # Test with invalid argument type (should raise ExpressionError)
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


# LLM-generated content at query #35
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
    assert str(result) == '/*[position() = 1]'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('Function', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('Arg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '/*[position() = 3]'

    # Test with invalid argument type (should raise ExpressionError)
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


# LLM-generated content at query #36
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
        assert str(e) == "Expected a single string or ident for :has(), got (NUMBER 123,)"


# LLM-generated content at query #37
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
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
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #38
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()
    xpath = translator.xpathexpr_cls()

    # Test with valid selector
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': '.bar'})],
        'argument_types': lambda: ['STRING']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::*[self::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]]'

    # Test with invalid selector type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 123})],
        'argument_types': lambda: ['NUMBER']
    })()
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(xpath, function)

    # Test with IDENT type
    function = type('Function', (), {
        'arguments': [type('Argument', (), {'value': 'div'})],
        'argument_types': lambda: ['IDENT']
    })()
    result = translator.xpath_has_function(xpath, function)
    assert result.post_condition == 'descendant::div'


# LLM-generated content at query #39
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
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert str(e) == "Expected a single string or ident for :contains(), got [NUMBER(123)]"


# LLM-generated content at query #40
#--------------------------

```python
def test_JQueryTranslator_xpath_gt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
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
    with pytest.raises(ExpressionError):
        translator.xpath_gt_function(xpath, function)


# LLM-generated content at query #41
#--------------------------

```python
def test_JQueryTranslator_xpath_has_function():
    translator = JQueryTranslator()

    # Test with matching selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {'arguments': [type('Argument', (), {'value': '.bar'})], 'argument_types': lambda: ['STRING']})()
    )
    assert str(xpath) == 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'

    # Test with non-matching selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {'arguments': [type('Argument', (), {'value': '.baz'})], 'argument_types': lambda: ['STRING']})()
    )
    assert str(xpath) == 'descendant::*[contains(concat(" ", @class, " "), " baz ")]'

    # Test with element selector
    xpath = translator.xpath_has_function(
        translator.xpathexpr_cls(),
        type('Function', (), {'arguments': [type('Argument', (), {'value': 'div'})], 'argument_types': lambda: ['STRING']})()
    )
    assert str(xpath) == 'descendant::div'

    # Test with invalid argument type
    with pytest.raises(ExpressionError):
        translator.xpath_has_function(
            translator.xpathexpr_cls(),
            type('Function', (), {'arguments': [type('Argument', (), {'value': 123})], 'argument_types': lambda: ['NUMBER']})()
        )


# LLM-generated content at query #42
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
        assert "Expected a single integer for :gt()" in str(e)


# LLM-generated content at query #43
#--------------------------

```python
def test_JQueryTranslator_xpath_eq_function():
    translator = JQueryTranslator()

    # Test with valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '0'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*'  # Default element
    assert result.post_condition == 'position() = 1'

    # Test with another valid integer argument
    xpath = translator.xpathexpr_cls()
    function = type('MockFunction', (), {
        'argument_types': lambda self: ['NUMBER'],
        'arguments': [type('MockArg', (), {'value': '2'})()]
    })()
    result = translator.xpath_eq_function(xpath, function)
    assert str(result) == '*'  # Default element
    assert result.post_condition == 'position() = 3'

    # Test with invalid argument type (should raise ExpressionError)
    xpath = translator.xpathexpr_cls()
    function = type('MockFunction', (), {
        'argument_types': lambda self: ['STRING'],
        'arguments': [type('MockArg', (), {'value': 'invalid'})()]
    })()
    try:
        translator.xpath_eq_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError as e:
        assert "Expected a single integer for :eq()" in str(e)


# LLM-generated content at query #44
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
        'arguments': [type('Argument', (), {'value': '123'})()]
    })()
    try:
        translator.xpath_contains_function(xpath, function)
        assert False, "Expected ExpressionError"
    except ExpressionError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_JQueryTranslator_xpath_lt_function():
    translator = JQueryTranslator()
    xpath = XPathExpr()

    # Test with valid number argument
    function = MockFunction(['NUMBER'], [MockArgument('0')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 1'

    # Test with another valid number argument
    function = MockFunction(['NUMBER'], [MockArgument('2')])
    result = translator.xpath_lt_function(xpath, function)
    assert result.post_condition == 'position() < 3'

    # Test with invalid argument type
    function = MockFunction(['STRING'], [MockArgument('"test"')])
    with pytest.raises(ExpressionError):
        translator.xpath_lt_function(xpath, function)


