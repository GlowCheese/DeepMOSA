####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_eq_function_valid_input. Retrieved 4/23 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_type. Retrieved 4/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = 'NUMBER'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = 'STRING'
    var_3 = [var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_has_function_with_valid_string_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_with_valid_ident_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_raises_error_on_invalid_argument_types. Retrieved 5/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '.bar'
    var_2 = 'STRING'
    var_3 = 'descendant::*[@class="bar"]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'IDENT'
    var_3 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'NUMBER'
    var_3 = 'ExpressionError not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 4/24 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/24 statements.
# Partially parsed test_xpath_contains_function_invalid_type. Retrieved 6/28 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "'title'"
    var_2 = 'STRING'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'IDENT'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 'ExpressionError not raised for invalid argument type'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_validation. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'not_a_number'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_has_function_valid_argument_types. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument_types. Retrieved 4/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = 'descendant::*[@class="bar"]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'div'
    var_3 = 'descendant::div'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_test_jquery_translator_init. Retrieved 1/3 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_gt_function_success. Retrieved 5/9 statements.
# Partially parsed test_xpath_gt_function_error_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'abc'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_on_non_number_argument. Retrieved 1/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 4/23 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'IDENT'
    var_3 = [var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_lt_function_valid_argument. Retrieved 3/12 statements.
# Partially parsed test_xpath_lt_function_invalid_argument_type. Retrieved 2/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'



# Parsed testcases at query #12
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.JQueryTranslator(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 123



# Parsed testcases at query #14
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.JQueryTranslator(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_has_function_valid_string_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_argument_types. Retrieved 5/15 statements.


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
    var_3 = 'ExpressionError not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_has_function_valid_argument_types. Retrieved 2/20 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '5'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_type_raises_error. Retrieved 3/13 statements.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_has_function_valid_argument_types. Retrieved 6/38 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = '.bar'
    var_3 = 'IDENT'
    var_4 = [var_3]
    var_5 = 'div'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_type. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_has_function_with_valid_string. Retrieved 5/12 statements.
# Partially parsed test_xpath_has_function_with_valid_ident. Retrieved 5/12 statements.
# Partially parsed test_xpath_has_function_with_invalid_type_raises_error. Retrieved 4/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '.bar'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'descendant::.bar'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '123'
    var_2 = 'NUMBER'
    var_3 = [var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 8/24 statements.
# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'test'
    var_3 = "'test'"
    var_4 = "contains(., 'test')"
    var_5 = 'IDENT'
    var_6 = 'title'
    var_7 = 'contains(., title)'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 123



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_eq_function_valid_input. Retrieved 3/12 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_type. Retrieved 2/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_lt_function_valid_input. Retrieved 4/12 statements.
# Partially parsed test_xpath_lt_function_invalid_argument_type. Retrieved 3/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '1'
    var_3 = 'position() < 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'not_a_number'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/17 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = 'abc'
    var_6 = 'STRING'
    var_7 = [var_6]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_eq_function_valid_integer. Retrieved 5/17 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 5/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 'NUMBER'
    var_4 = [var_3]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = [var_1]
    var_3 = 'STRING'
    var_4 = [var_3]



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_xpath_gt_function_argument_types_valid.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_gt_function_valid. Retrieved 4/23 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 4/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '1'
    var_2 = 'NUMBER'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = 'STRING'
    var_3 = [var_2]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 5/17 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '123'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 3/28 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_types. Retrieved 5/30 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'

def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'not_a_number'
    var_3 = 'Should have raised an exception'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 4/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_image_pseudo. Retrieved 2/7 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 4/9 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/9 statements.
# Partially parsed test_xpath_contains_function_invalid_type_list. Retrieved 3/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = lambda self: ['STRING']
    var_3 = "'%s'"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = lambda self: ['IDENT']
    var_3 = "'%s'"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = lambda self: ['NUMBER']



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_eq_function_valid. Retrieved 4/23 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 4/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = 'NUMBER'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = 'STRING'
    var_3 = [var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_gt_function_valid_input. Retrieved 3/12 statements.
# Partially parsed test_xpath_gt_function_invalid_argument_type. Retrieved 2/10 statements.
# Partially parsed test_xpath_gt_function_different_index. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 6'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 3/28 statements.
# Partially parsed test_xpath_contains_function_valid_ident_type. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'title'

def test_case_0():
    var_0 = 'IDENT'
    var_1 = [var_0]
    var_2 = 'title'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_type_raises_error. Retrieved 5/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = 'descendant::*[contains(@class, "bar")]'

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
    var_3 = 'ExpressionError not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_has_function_valid_string_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_valid_ident_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_invalid_argument_type_raises_error. Retrieved 5/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '.bar'
    var_2 = 'STRING'
    var_3 = "descendant::*[@class='bar']"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'IDENT'
    var_3 = 'descendant::div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'NUMBER'
    var_3 = 'ExpressionError not raised for invalid argument type'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_gt_function_valid. Retrieved 5/16 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 2/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = 'NUMBER'
    var_4 = [var_3]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 7/32 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'
    var_4 = 'STRING'
    var_5 = [var_4]
    var_6 = 'abc'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/17 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '1'
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = 'abc'
    var_6 = 'STRING'
    var_7 = [var_6]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_is_number. Retrieved 1/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_has_function_invalid_argument_types_raises_expression_error. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 2/22 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 123
    var_3 = 'The predicate at line 11 should have evaluated to True, causing an exception.'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 123



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_gt_function_valid_argument_types. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_eq_function_success. Retrieved 5/9 statements.
# Partially parsed test_xpath_eq_function_error_non_numeric. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'abc'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_eq_function_valid_number_type. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_eq_function_valid_number_type. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_has_function_valid_arguments. Retrieved 19/29 statements.
# Partially parsed test_xpath_has_function_ident_arguments. Retrieved 19/29 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPath'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, x: var_4
    var_6 = {var_3: var_5}
    var_7 = 'Function'
    var_8 = ()
    var_9 = 'argument_types'
    var_10 = 'arguments'
    var_11 = 'STRING'
    var_12 = [var_11]
    var_13 = lambda self: var_12
    var_14 = 'Argument'
    var_15 = ()
    var_16 = 'value'
    var_17 = '.bar'
    var_18 = {var_16: var_17}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'XPath'
    var_2 = ()
    var_3 = 'add_post_condition'
    var_4 = None
    var_5 = lambda self, x: var_4
    var_6 = {var_3: var_5}
    var_7 = 'Function'
    var_8 = ()
    var_9 = 'argument_types'
    var_10 = 'arguments'
    var_11 = 'IDENT'
    var_12 = [var_11]
    var_13 = lambda self: var_12
    var_14 = 'Argument'
    var_15 = ()
    var_16 = 'value'
    var_17 = 'div'
    var_18 = {var_16: var_17}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_is_number. Retrieved 3/30 statements.
# Partially parsed test_xpath_eq_function_raises_error_on_wrong_type. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'

def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = '"not a number"'
    var_3 = 'Exception should have been raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 4/22 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/26 statements.
# Partially parsed test_xpath_contains_function_invalid_type. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'IDENT'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 123
    var_2 = 'NUMBER'
    var_3 = [var_2]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 5/37 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'title'
    var_3 = 'IDENT'
    var_4 = [var_3]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 5/10 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 5/10 statements.
# Partially parsed test_xpath_has_function_invalid_type_list. Retrieved 5/11 statements.
# Partially parsed test_xpath_has_function_invalid_type_tuple. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '.baz'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'BOOLEAN'
    var_3 = [var_2]
    var_4 = 'true'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 5/10 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 5/10 statements.
# Partially parsed test_xpath_has_function_invalid_type_list. Retrieved 7/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '.bar'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '123'
    var_5 = 'ExpressionError not raised'
    var_6 = AssertionError(var_5)



