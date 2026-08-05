####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_lt_function_valid. Retrieved 4/12 statements.
# Partially parsed test_xpath_lt_function_invalid_type. Retrieved 3/13 statements.


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
    var_2 = 'abc'
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_gt_function_valid. Retrieved 5/9 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'
    var_5 = 'position() > 1'
    var_6 = bool('position() > 1' in var_1.post_conditions)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'abc'
    var_5 = 'Expected a single integer for :gt(), got'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 4/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_eq_function_valid_index. Retrieved 4/21 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 4/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = []
    var_4 = bool(str(e).startswith('Expected a single integer'))
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = 'position() = 1'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() > 1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 28/29 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 28/29 statements.


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
    var_17 = lambda self: var_16
    var_18 = 'MockArg'
    var_19 = ()
    var_20 = 'value'
    var_21 = '.bar'
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
    var_33 = var_0.xpath_has_function(var_10, var_32)
    var_34 = bool(var_33 == var_10)
    assert var_34 is True

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
    var_17 = lambda self: var_16
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
    var_32 = var_31()
    var_33 = var_0.xpath_has_function(var_10, var_32)
    var_34 = bool(var_33 == var_10)
    assert var_34 is True

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
    var_17 = lambda self: var_16
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
    var_32 = var_31()
    var_33 = var_0.xpath_has_function(var_10, var_32)
    var_34 = 'Expected a single string or ident'
    var_35 = 'ExpressionError not raised'
    var_36 = AssertionError(var_35)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 5/14 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 5/14 statements.
# Partially parsed test_xpath_contains_function_raises_error_on_invalid_type. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'title'
    var_3 = "'title'"
    var_4 = "contains(., 'title')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'title'
    var_3 = "'title'"
    var_4 = "contains(., 'title')"

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 123
    var_3 = 'Expected a single string or ident for :contains()'



# Parsed testcases at query #10
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
    var_5 = 'Expected a single integer for :gt()'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = bool(str(e).startswith('Expected a single string or ident for :contains()'))
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = 'Expected a single string or ident for :contains()'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 4/28 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'
    var_4 = 'position() < 2'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 2/23 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 2/23 statements.


def test_case_0():
    var_0 = 'title'
    var_1 = []
    var_2 = "'title'"

def test_case_0():
    var_0 = 'title'
    var_1 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = 'position() = 1'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_gt_function_valid_argument. Retrieved 4/12 statements.
# Partially parsed test_xpath_gt_function_invalid_argument_type. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = 'position() > 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'not_a_number'
    var_3 = 'Expected a single integer for :gt()'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 5/20 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Args'
    var_2 = 'value'
    var_3 = '1'
    var_4 = {var_2: var_3}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_has_function_invalid_argument_types_raises_error. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'ExpressionError was not raised for invalid argument types'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 2/22 statements.
# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 2/18 statements.
# Partially parsed test_xpath_contains_function_ident_type. Retrieved 2/22 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 'Expected a single string or ident'

def test_case_0():
    var_0 = 'IDENT'
    var_1 = [var_0]
    var_2 = "contains(., 'title')"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 123
    var_3 = bool(str(e).startswith('Expected a single string or ident for :contains()'))
    assert var_3 is True
    var_4 = 'Predicate at line 11 should have evaluated to True to trigger the exception'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_has_function_argument_types_valid. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = '.bar'
    var_3 = 'descendant::.bar'



# Parsed testcases at query #25
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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_first_pseudo. Retrieved 2/6 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'position() = 1'



# Parsed testcases at query #2
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = var_0.xpath_header_pseudo(var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True
    var_4 = "(name(.) = 'h1' or name(.) = 'h2' or name (.) = 'h3') or (name(.) = 'h4' or name (.) = 'h5' or name(.) = 'h6')"
    var_5 = bool("(name(.) = 'h1' or name(.) = 'h2' or name (.) = 'h3') or (name(.) = 'h4' or name (.) = 'h5' or name(.) = 'h6')" in var_1.conditions)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 9/36 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'
    var_4 = 'STRING'
    var_5 = [var_4]
    var_6 = 'abc'
    var_7 = [var_1]
    var_8 = '0'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 5/9 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 5/9 statements.
# Partially parsed test_xpath_has_function_invalid_type_number. Retrieved 5/10 statements.
# Partially parsed test_xpath_has_function_invalid_type_list. Retrieved 5/10 statements.


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
    var_5 = 'Expected a single string or ident'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'LIST'
    var_3 = [var_2]
    var_4 = '["a"]'
    var_5 = 'Expected a single string or ident'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_gt_function_valid_number. Retrieved 4/12 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '1'
    var_3 = 'position() > 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'not_a_number'
    var_3 = 'Expected a single integer for :gt()'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument. Retrieved 3/12 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_type. Retrieved 3/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'foo'
    var_3 = 'Expected a single integer for :eq(), got'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 5/11 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 6/13 statements.
# Partially parsed test_xpath_contains_function_invalid_type_list. Retrieved 4/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = '"title"'
    var_4 = 'contains(., "title")'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = [var_1]
    var_3 = 'title'
    var_4 = '"title"'
    var_5 = 'contains(., "title")'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = 123
    var_4 = 'Expected a single string or ident for :contains()'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_validation. Retrieved 6/39 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'abc'
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = 'position() = 1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_eq_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_eq_function_validates_number_type. Retrieved 4/29 statements.


def test_case_0():
    var_0 = None
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() = 1'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 4/23 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 4/23 statements.
# Partially parsed test_xpath_contains_function_invalid_type. Retrieved 4/20 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = "contains(., 'title')"

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
    var_4 = 'Expected a single string or ident'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_gt_function_valid. Retrieved 4/15 statements.
# Partially parsed test_xpath_gt_function_invalid_type. Retrieved 2/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0.JQueryTranslator()
    var_3 = '1'
    var_4 = 'position() > 2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'foo'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_has_function_valid_string. Retrieved 5/9 statements.
# Partially parsed test_xpath_has_function_valid_ident. Retrieved 5/9 statements.
# Partially parsed test_xpath_has_function_invalid_type_number. Retrieved 5/11 statements.
# Partially parsed test_xpath_has_function_invalid_type_list. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '.baz'
    var_5 = 'descendant::.baz'
    var_6 = bool('descendant::.baz' in var_1.post_conditions)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'div'
    var_5 = 'descendant::div'
    var_6 = bool('descendant::div' in var_1.post_conditions)
    assert var_6 is True

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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/23 statements.


def test_case_0():
    var_0 = '0'
    var_1 = 'position() > 1'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument_types. Retrieved 6/22 statements.
# Partially parsed test_xpath_contains_function_invalid_argument_types. Retrieved 6/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'test'
    var_3 = "'test'"
    var_4 = "contains(., 'test')"
    var_5 = 'IDENT'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 123
    var_3 = 'Expected a single string or ident'
    var_4 = 'STRING'
    var_5 = 'IDENT'
    var_6 = 'test'
    var_7 = 'Expected a single string or ident'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_contains_function_raises_error_on_invalid_argument_types. Retrieved 5/28 statements.
# Partially parsed test_xpath_contains_function_works_with_valid_string_type. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = 'Expected a single string or ident for :contains()'
    var_4 = 'The predicate at line 11 should have evaluated to True, causing an exception.'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]
    var_2 = 'test'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/29 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '1'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = 'abc'
    var_8 = 'Expected a single integer'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_has_function. Retrieved 10/29 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.baz'
    var_3 = 'descendant::*[@class="baz"]'
    var_4 = 'IDENT'
    var_5 = 'div'
    var_6 = 'descendant::div'
    var_7 = 'NUMBER'
    var_8 = 'ExpressionError not raised for invalid argument type'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_has_function_invalid_argument_types. Retrieved 2/19 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = 'Expected a single string or ident for :has()'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() = 1'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_has_function_with_valid_string_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_with_valid_ident_argument. Retrieved 4/13 statements.
# Partially parsed test_xpath_has_function_raises_error_on_invalid_argument_type. Retrieved 3/12 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument. Retrieved 4/23 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_type. Retrieved 4/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'Expected a single integer for :eq(), got'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/20 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 4/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'
    var_4 = 'position() < 2'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_eq_function_valid_argument_types. Retrieved 4/29 statements.


def test_case_0():
    var_0 = None
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string. Retrieved 5/9 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 5/9 statements.
# Partially parsed test_xpath_contains_function_invalid_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"title"'
    var_5 = 'contains(., "title")'
    var_6 = bool('contains(., "title")' in var_1.post_conditions)
    assert var_6 is True

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
    var_4 = '123'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_eq_function_valid. Retrieved 5/17 statements.
# Partially parsed test_xpath_eq_function_invalid_type. Retrieved 5/17 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = 'position() = 1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'abc'
    var_2 = [var_1]
    var_3 = 'STRING'
    var_4 = [var_3]



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 3/24 statements.


def test_case_0():
    var_0 = 'NUMBER'
    var_1 = [var_0]
    var_2 = '1'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 1/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



