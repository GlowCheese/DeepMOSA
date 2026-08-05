# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.cssselectpatch as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_button_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.JQueryTranslator()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_0.xhtml is False
    assert var_0.lower_case_element_names is True
    assert var_0.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = None
    var_0.xpath_password_pseudo(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = None
    var_2 = var_0.add_post_condition(var_1)
    var_2.xpath_enabled_pseudo(var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_text_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_last_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_checked_pseudo(var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.JQueryTranslator(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is None
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2 = module_0.XPathExpr()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_2.path == ''
    assert var_2.element == '*'
    assert var_2.condition == ''
    assert var_2.post_condition is None
    var_3 = var_1.xpath_enabled_pseudo(var_2)
    assert var_2.condition == "(\n            ((name(.) = 'button' or name(.) = 'input' or name(.) = 'select'\n                    or name(.) = 'textarea' or name(.) = 'fieldset')\n                and not(@disabled or (ancestor::fieldset[@disabled]\n                    and not(ancestor::legend[not(preceding-sibling::legend)])))\n            )\n            or\n            ((name(.) = 'option'\n                and not(@disabled or ancestor::optgroup[@disabled]))\n            )\n            or\n            ((name(.) = 'optgroup' and not(@disabled)))\n            )"
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_3.path == ''
    assert var_3.element == '*'
    assert var_3.condition == "(\n            ((name(.) = 'button' or name(.) = 'input' or name(.) = 'select'\n                    or name(.) = 'textarea' or name(.) = 'fieldset')\n                and not(@disabled or (ancestor::fieldset[@disabled]\n                    and not(ancestor::legend[not(preceding-sibling::legend)])))\n            )\n            or\n            ((name(.) = 'option'\n                and not(@disabled or ancestor::optgroup[@disabled]))\n            )\n            or\n            ((name(.) = 'optgroup' and not(@disabled)))\n            )"
    assert var_3.post_condition is None

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_image_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = None
    var_2 = module_0.JQueryTranslator(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_2.xhtml is None
    assert var_2.lower_case_element_names is True
    assert var_2.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2.xpath_file_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_header_pseudo(var_0)

def test_case_11():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = var_0.__str__()
    assert var_1 == '*'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_0.join(var_1, var_1, has_inner_condition=var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.JQueryTranslator(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is None
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2 = module_0.XPathExpr()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_2.path == ''
    assert var_2.element == '*'
    assert var_2.condition == ''
    assert var_2.post_condition is None
    var_1.xpath_odd_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_reset_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = None
    var_2 = module_0.JQueryTranslator(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_2.xhtml is None
    assert var_2.lower_case_element_names is True
    assert var_2.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2.xpath_even_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.XPathExpr(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_1.path is None
    assert var_1.element == '*'
    assert var_1.condition == ''
    assert var_1.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2 = var_1.__repr__()
    assert var_2 == 'XPathExpr[None*]'
    var_3 = module_0.JQueryTranslator()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_3.xhtml is False
    assert var_3.lower_case_element_names is True
    assert var_3.lower_case_attribute_names is True
    var_3.xpath_checkbox_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = var_0.add_name_test()
    var_2 = var_0.__str__()
    assert var_2 == '*'
    var_3 = var_0.add_post_condition(var_0)
    assert f'{type(var_0.post_condition).__module__}.{type(var_0.post_condition).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    var_0.join(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = var_0.__repr__()
    assert var_1 == 'XPathExpr[*]'
    var_2 = var_0.add_post_condition(var_1)
    assert var_0.post_condition == 'XPathExpr[*]'
    var_3 = None
    var_4 = var_0.add_post_condition(var_3)
    assert var_0.post_condition == 'XPathExpr[*] and (None)'
    var_0.join(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = var_0.__repr__()
    assert var_1 == 'XPathExpr[*]'
    var_2 = None
    var_3 = module_0.JQueryTranslator()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_3.xhtml is False
    assert var_3.lower_case_element_names is True
    assert var_3.lower_case_attribute_names is True
    var_3.xpath_hidden_pseudo(var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_empty_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = None
    var_2 = module_0.JQueryTranslator(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_2.xhtml is None
    assert var_2.lower_case_element_names is True
    assert var_2.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2.xpath_radio_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.JQueryTranslator()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_0.xhtml is False
    assert var_0.lower_case_element_names is True
    assert var_0.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = None
    var_0.xpath_selected_pseudo(var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = None
    var_2 = True
    var_3 = module_0.JQueryTranslator(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_3.xhtml is True
    var_3.xpath_parent_pseudo(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = None
    var_2 = module_0.JQueryTranslator()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_2.xhtml is False
    assert var_2.lower_case_element_names is True
    assert var_2.lower_case_attribute_names is True
    var_2.xpath_first_pseudo(var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = False
    var_1 = module_0.JQueryTranslator(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_2 = module_0.XPathExpr()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_2.path == ''
    assert var_2.element == '*'
    assert var_2.condition == ''
    assert var_2.post_condition is None
    var_3 = None
    var_1.xpath_disabled_pseudo(var_3)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_input_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.JQueryTranslator()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_1.xhtml is False
    assert var_1.lower_case_element_names is True
    assert var_1.lower_case_attribute_names is True
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1.xpath_submit_pseudo(var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_0.XPathExpr()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_0.path == ''
    assert var_0.element == '*'
    assert var_0.condition == ''
    assert var_0.post_condition is None
    assert f'{type(module_0.unicode_literals).__module__}.{type(module_0.unicode_literals).__qualname__}' == '__future__._Feature'
    assert module_0.unicode_literals.optional == (2, 6, 0, 'alpha', 2)
    assert module_0.unicode_literals.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.unicode_literals.compiler_flag == 2097152
    var_1 = var_0.__repr__()
    assert var_1 == 'XPathExpr[*]'
    var_2 = None
    var_3 = False
    var_4 = module_0.JQueryTranslator(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyquery.cssselectpatch.JQueryTranslator'
    assert var_4.xhtml is False
    assert var_4.lower_case_element_names is True
    assert var_4.lower_case_attribute_names is True
    var_5 = var_4.xpath_disabled_pseudo(var_0)
    assert var_0.condition == "(\n            ((name(.) = 'button' or name(.) = 'input' or name(.) = 'select'\n                    or name(.) = 'textarea' or name(.) = 'fieldset')\n                and (@disabled or (ancestor::fieldset[@disabled]\n                    and not(ancestor::legend[not(preceding-sibling::legend)])))\n            )\n            or\n            ((name(.) = 'option'\n                and (@disabled or ancestor::optgroup[@disabled]))\n            )\n            or\n            ((name(.) = 'optgroup' and (@disabled)))\n            )"
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyquery.cssselectpatch.XPathExpr'
    assert var_5.path == ''
    assert var_5.element == '*'
    assert var_5.condition == "(\n            ((name(.) = 'button' or name(.) = 'input' or name(.) = 'select'\n                    or name(.) = 'textarea' or name(.) = 'fieldset')\n                and (@disabled or (ancestor::fieldset[@disabled]\n                    and not(ancestor::legend[not(preceding-sibling::legend)])))\n            )\n            or\n            ((name(.) = 'option'\n                and (@disabled or ancestor::optgroup[@disabled]))\n            )\n            or\n            ((name(.) = 'optgroup' and (@disabled)))\n            )"
    assert var_5.post_condition is None
    var_5.__setitem__(var_2, var_2)