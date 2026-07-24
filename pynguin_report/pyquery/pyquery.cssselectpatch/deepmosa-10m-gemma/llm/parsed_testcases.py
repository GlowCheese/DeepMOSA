####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 4/22 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 4/23 statements.
# Partially parsed test_xpath_contains_function_raises_error_on_invalid_type. Retrieved 6/24 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "'test'"
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = "contains(., 'test')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = "contains(., 'title')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 'ExpressionError not raised for invalid argument type'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_lt_function_valid. Retrieved 5/9 statements.
# Partially parsed test_xpath_lt_function_invalid_type. Retrieved 5/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '1'
    var_5 = 'position() < 2'
    var_6 = bool('position() < 2' in var_1.post_conditions)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'foo'
    var_5 = 'Expected a single integer for :gt(), got'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'
    var_3 = 'position() < 2'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_type_raises_error. Retrieved 3/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.baz'
    var_3 = 'descendant::*[@class="baz"]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'div'
    var_3 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '123'
    var_3 = 'Expected a single string or ident'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 2/22 statements.
# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 2/18 statements.
# Partially parsed test_xpath_contains_function_valid_ident_type. Retrieved 2/20 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'IDENT'
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_lt_function_valid. Retrieved 2/20 statements.
# Partially parsed test_xpath_lt_function_invalid_type. Retrieved 2/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '1'
    var_2 = 'position() < 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = 'Expected a single integer'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 10/30 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '1'
    var_5 = [var_2]
    var_6 = '0'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = '"text"'
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_valid_input. Retrieved 5/23 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 3/22 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '0'
    var_3 = 'NUMBER'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'abc'
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = 'Expected a single integer for :eq()'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 5/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'
    var_5 = var_1.post_conditions
    var_6 = bool(var_1.post_conditions == ['position() > 1'])
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 5/38 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'title'
    var_3 = 'IDENT'
    var_4 = [var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = 'position() = 1'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = var_0.xpath_image_pseudo(var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True
    var_4 = "@type = 'image' and name(.) = 'input'"
    var_5 = bool("@type = 'image' and name(.) = 'input'" in var_1.conditions)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_password_pseudo. Retrieved 2/6 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "@type = 'password' and name(.) = 'input'"



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_eq_function_valid. Retrieved 3/12 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_xpath_eq_function_different_index. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 6'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_test_jquery_translator_init. Retrieved 1/4 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #5
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.JQueryTranslator(var_0)
    var_2 = var_1.lower_case_attribute_names
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_lt_function_valid. Retrieved 4/16 statements.
# Partially parsed test_xpath_lt_function_invalid_type. Retrieved 5/18 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '1'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 'position() < 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = 'Should have raised ExpressionError'
    var_4 = AssertionError(var_3)
    var_5 = bool(str(e).startswith('Expected a single integer'))
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_gt_function_valid. Retrieved 4/21 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 5/18 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '1'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 'position() > 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = None
    var_2 = [var_1]
    var_3 = 'STRING'
    var_4 = [var_3]
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 5/11 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 5/11 statements.
# Partially parsed test_xpath_contains_function_invalid_type_int. Retrieved 4/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "'target'"
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = "contains(., 'target')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'some_id'
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'contains(., some_id)'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '123'
    var_2 = 'NUMBER'
    var_3 = [var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 5/17 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/16 statements.
# Partially parsed test_xpath_contains_function_invalid_type_raises_error. Retrieved 5/18 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'title'
    var_3 = 'STRING'
    var_4 = [var_3]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 123
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_eq_function_valid_number. Retrieved 4/12 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 5/15 statements.
# Partially parsed test_xpath_eq_function_large_index. Retrieved 4/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'abc'
    var_3 = 'Expected a single integer for :eq(), got'
    var_4 = 'ExpressionError not raised'
    var_5 = AssertionError(var_4)

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '5'
    var_3 = 'position() = 6'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 5/9 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 5/9 statements.
# Partially parsed test_xpath_contains_function_invalid_type_list. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"title"'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 123



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_is_number. Retrieved 3/29 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'not_a_number'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_type_raises_error. Retrieved 5/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.baz'
    var_3 = 'descendant::*[@class="baz"]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'div'
    var_3 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '123'
    var_3 = 'ExpressionError not raised for invalid argument type'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 8/24 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '"title"'
    var_3 = "'title'"
    var_4 = "contains(., 'title')"
    var_5 = 'IDENT'
    var_6 = 'title'
    var_7 = 'contains(., title)'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_lt_function_valid. Retrieved 5/9 statements.
# Partially parsed test_xpath_lt_function_invalid_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '1'
    var_5 = 'position() < 2'
    var_6 = bool('position() < 2' in var_1.post_conditions)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'abc'



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_xpath_lt_function_argument_types_is_not_number.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_eq_function_success. Retrieved 7/13 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'
    var_5 = var_1.post_conditions
    var_6 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'foo'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 5/34 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'IDENT'
    var_4 = [var_3]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_has_function_valid_string_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_argument_type_raises_error. Retrieved 3/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.child'
    var_3 = 'descendant::*[@class="child"]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'div'
    var_3 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '123'
    var_3 = 'Expected a single string or ident'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 7/23 statements.
# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = "'title'"
    var_3 = "contains(., 'title')"
    var_4 = 'IDENT'
    var_5 = 'title'
    var_6 = 'contains(., title)'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 123
    var_3 = 'Expected a single string or ident'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_xpath_gt_function_argument_types_is_not_number.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_has_function_valid_string_argument. Retrieved 3/13 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = '.bar'
    var_2 = 'descendant::div'

def test_case_0():
    var_0 = 'IDENT'
    var_1 = 'div'
    var_2 = 'descendant::div'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = bool(str(e).startswith('Expected a single string or ident for :contains()'))
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_is_number. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() = 1'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_has_function_argument_types_valid. Retrieved 9/45 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = '.bar'
    var_3 = 'IDENT'
    var_4 = [var_3]
    var_5 = 'div'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = '1'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 23/24 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 24/26 statements.
# Partially parsed test_xpath_has_function_invalid_type_raises_error. Retrieved 26/30 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPathMock'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'FunctionMock'
    var_12 = ()
    var_13 = 'arguments'
    var_14 = 'Arg'
    var_15 = ()
    var_16 = 'value'
    var_17 = '.baz'
    var_18 = {var_16: var_17}
    var_19 = [var_14, var_15, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = [var_21]
    var_23 = {var_13: var_22}
    var_24 = [var_11, var_12, var_23]
    var_25 = {}
    var_26 = module_1.type(*var_24, **var_25)
    var_27 = var_26()
    var_28 = var_0.xpath_has_function(var_10, var_27)
    var_29 = bool(var_28 == var_10)
    assert var_29 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPathMock'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'FunctionMock'
    var_12 = ()
    var_13 = 'arguments'
    var_14 = 'Arg'
    var_15 = ()
    var_16 = 'value'
    var_17 = 'div'
    var_18 = {var_16: var_17}
    var_19 = [var_14, var_15, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = [var_21]
    var_23 = {var_13: var_22}
    var_24 = [var_11, var_12, var_23]
    var_25 = {}
    var_26 = module_1.type(*var_24, **var_25)
    var_27 = var_26()
    var_28 = 'IDENT'
    var_29 = var_0.xpath_has_function(var_10, var_27)
    var_30 = bool(var_29 == var_10)
    assert var_30 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPathMock'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'FunctionMock'
    var_12 = ()
    var_13 = 'arguments'
    var_14 = 'Arg'
    var_15 = ()
    var_16 = 'value'
    var_17 = 123
    var_18 = {var_16: var_17}
    var_19 = [var_14, var_15, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = [var_21]
    var_23 = {var_13: var_22}
    var_24 = [var_11, var_12, var_23]
    var_25 = {}
    var_26 = module_1.type(*var_24, **var_25)
    var_27 = var_26()
    var_28 = 'NUMBER'
    var_29 = var_0.xpath_has_function(var_10, var_27)
    var_30 = 'ExpressionError not raised for invalid argument type'
    var_31 = AssertionError(var_30)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'
    var_3 = 'position() < 2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_gt_function_logic. Retrieved 19/26 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPathMock'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'FunctionMock'
    var_12 = ()
    var_13 = 'argument_types'
    var_14 = 'arguments'
    var_15 = 'NUMBER'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = 'ArgumentMock'
    var_19 = ()
    var_20 = 'value'
    var_21 = '0'
    var_22 = {var_20: var_21}
    var_23 = [var_18, var_19, var_22]
    var_24 = {}
    var_25 = module_1.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = [var_26]
    var_28 = {var_13: var_17, var_14: var_27}
    var_29 = [var_11, var_12, var_28]
    var_30 = {}
    var_31 = module_1.type(*var_29, **var_30)
    var_32 = var_31()
    var_33 = var_0.xpath_gt_function(var_10, var_32)

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPathMock'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'FunctionMock'
    var_12 = ()
    var_13 = 'argument_types'
    var_14 = 'arguments'
    var_15 = 'STRING'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = 'ArgumentMock'
    var_19 = ()
    var_20 = 'value'
    var_21 = 'abc'
    var_22 = {var_20: var_21}
    var_23 = [var_18, var_19, var_22]
    var_24 = {}
    var_25 = module_1.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = [var_26]
    var_28 = {var_13: var_17, var_14: var_27}
    var_29 = [var_11, var_12, var_28]
    var_30 = {}
    var_31 = module_1.type(*var_29, **var_30)
    var_32 = var_31()
    var_33 = ()
    var_34 = 'NUMBER'
    var_35 = [var_34]
    var_36 = lambda self: var_35
    var_37 = ()
    var_38 = '2'
    var_39 = {var_20: var_38}
    var_40 = [var_18, var_37, var_39]
    var_41 = {}
    var_42 = module_1.type(*var_40, **var_41)
    var_43 = var_42()
    var_44 = [var_43]
    var_45 = {var_13: var_36, var_14: var_44}
    var_46 = [var_11, var_33, var_45]
    var_47 = {}
    var_48 = module_1.type(*var_46, **var_47)
    var_49 = var_48()
    var_50 = var_0.xpath_gt_function(var_10, var_49)

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'FunctionMock'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'ArgumentMock'
    var_9 = ()
    var_10 = 'value'
    var_11 = '5'
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = [var_16]
    var_18 = {var_3: var_7, var_4: var_17}
    var_19 = [var_1, var_2, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_has_function_argument_types_valid. Retrieved 3/30 statements.
# Partially parsed test_xpath_has_function_argument_types_invalid. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = '.bar'

def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 'Should have raised ExpressionError'
    var_3 = AssertionError(var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_has_function_valid_string_argument. Retrieved 27/28 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument. Retrieved 27/28 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockXPath'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'MockFunction'
    var_12 = ()
    var_13 = 'argument_types'
    var_14 = 'arguments'
    var_15 = 'STRING'
    var_16 = [var_15]
    var_17 = lambda : var_16
    var_18 = 'MockArg'
    var_19 = ()
    var_20 = 'value'
    var_21 = '.child'
    var_22 = {var_20: var_21}
    var_23 = [var_18, var_19, var_22]
    var_24 = {}
    var_25 = module_1.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = [var_26]
    var_28 = {var_13: var_17, var_14: var_27}
    var_29 = [var_11, var_12, var_28]
    var_30 = {}
    var_31 = module_1.type(*var_29, **var_30)
    var_32 = var_0.xpath_has_function(var_10, var_31)
    var_33 = bool(var_32 == var_10)
    assert var_33 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockXPath'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'MockFunction'
    var_12 = ()
    var_13 = 'argument_types'
    var_14 = 'arguments'
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda : var_16
    var_18 = 'MockArg'
    var_19 = ()
    var_20 = 'value'
    var_21 = 'div'
    var_22 = {var_20: var_21}
    var_23 = [var_18, var_19, var_22]
    var_24 = {}
    var_25 = module_1.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = [var_26]
    var_28 = {var_13: var_17, var_14: var_27}
    var_29 = [var_11, var_12, var_28]
    var_30 = {}
    var_31 = module_1.type(*var_29, **var_30)
    var_32 = var_0.xpath_has_function(var_10, var_31)
    var_33 = bool(var_32 == var_10)
    assert var_33 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockXPath'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, cond: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'MockFunction'
    var_12 = ()
    var_13 = 'argument_types'
    var_14 = 'arguments'
    var_15 = 'NUMBER'
    var_16 = [var_15]
    var_17 = lambda : var_16
    var_18 = 'MockArg'
    var_19 = ()
    var_20 = 'value'
    var_21 = '123'
    var_22 = {var_20: var_21}
    var_23 = [var_18, var_19, var_22]
    var_24 = {}
    var_25 = module_1.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = [var_26]
    var_28 = {var_13: var_17, var_14: var_27}
    var_29 = [var_11, var_12, var_28]
    var_30 = {}
    var_31 = module_1.type(*var_29, **var_30)
    var_32 = var_0.xpath_has_function(var_10, var_31)
    var_33 = 'Expected a single string or ident'
    var_34 = 'ExpressionError not raised'
    var_35 = AssertionError(var_34)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 3/14 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/23 statements.
# Partially parsed test_xpath_contains_function_invalid_type. Retrieved 2/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = 'test_text'
    var_3 = "contains(., 'test_text')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = 'ident_val'
    var_3 = 'IDENT'
    var_4 = "contains(., 'ident_val')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'Expected a single string or ident'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_eq_function_valid_integer. Retrieved 5/17 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 5/19 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]
    var_5 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = 'abc'
    var_4 = [var_3]
    var_5 = 'Expected a single integer'



