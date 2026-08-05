####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_contains_function_returns_xpath_with_contains_condition. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'text'
    var_13 = {var_11: var_12}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_has_function_matches_selector. Retrieved 7/11 statements.
# Partially parsed test_xpath_has_function_no_match. Retrieved 7/11 statements.
# Partially parsed test_xpath_has_function_with_ident. Retrieved 5/9 statements.
# Partially parsed test_xpath_has_function_raises_error_on_invalid_argument. Retrieved 6/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'class="foo"'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'has'
    var_5 = 'STRING'
    var_6 = '.bar'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'class="foo"'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'has'
    var_5 = 'STRING'
    var_6 = '.baz'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'has'
    var_4 = 'IDENT'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'has'
    var_4 = 'NUMBER'
    var_5 = '1'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_contains_function_passes_for_string_argument. Retrieved 5/12 statements.
# Partially parsed test_xpath_contains_function_passes_for_ident_argument. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'STRING'
    var_4 = '"title"'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'IDENT'
    var_4 = 'title'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 1/2 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_invalid_argument_types. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_gt_function_positive_index. Retrieved 14/21 statements.
# Partially parsed test_xpath_gt_function_negative_index. Retrieved 14/21 statements.
# Partially parsed test_xpath_gt_function_raises_error_for_non_number. Retrieved 14/22 statements.
# Partially parsed test_xpath_gt_function_raises_error_for_multiple_arguments. Retrieved 17/26 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '-1'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'test'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6, var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = '1'
    var_16 = {var_11: var_15}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_eq_function_single_number_argument. Retrieved 3/7 statements.
# Partially parsed test_xpath_eq_function_second_element. Retrieved 3/7 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_non_number_argument. Retrieved 16/29 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = 'MockFunction'
    var_1 = 'argument_types'
    var_2 = 'arguments'
    var_3 = 'STRING'
    var_4 = [var_3]
    var_5 = lambda self: var_4
    var_6 = 'MockArgument'
    var_7 = 'value'
    var_8 = 'not_a_number'
    var_9 = {var_7: var_8}
    var_10 = module_0.JQueryTranslator()
    var_11 = 'MockXPath'
    var_12 = 'add_post_condition'
    var_13 = None
    var_14 = lambda self, cond: var_13
    var_15 = {var_12: var_14}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_contains_function_with_string. Retrieved 14/22 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 14/22 statements.
# Partially parsed test_xpath_contains_function_raises_on_invalid_arguments. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'content'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '42'
    var_13 = {var_11: var_12}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_non_string_or_ident_argument. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_gt_function_accepts_number. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = (var_2,)
    var_4 = '0'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_lt_function_returns_condition_with_position_less_than_value_plus_one. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_with_zero_value. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_raises_error_on_non_number_argument. Retrieved 13/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = {var_11: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 4/10 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 4/10 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_type. Retrieved 4/11 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_multiple_arguments. Retrieved 5/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'STRING'
    var_3 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'IDENT'
    var_3 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'NUMBER'
    var_3 = 1

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'STRING'
    var_3 = 'a'
    var_4 = 'b'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_gt_function_with_valid_number. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_has_function_raises_for_invalid_argument_type. Retrieved 5/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_eq_function_with_number_argument. Retrieved 14/21 statements.
# Partially parsed test_xpath_eq_function_with_zero_index. Retrieved 14/21 statements.
# Partially parsed test_xpath_eq_function_with_negative_index_raises_error. Retrieved 14/22 statements.
# Partially parsed test_xpath_eq_function_with_non_number_argument_raises_error. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '2'
    var_10 = {var_8: var_9}
    var_11 = 'NUMBER'
    var_12 = [var_11]
    var_13 = lambda self: var_12

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}
    var_11 = 'NUMBER'
    var_12 = [var_11]
    var_13 = lambda self: var_12

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '-1'
    var_10 = {var_8: var_9}
    var_11 = 'NUMBER'
    var_12 = [var_11]
    var_13 = lambda self: var_12

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'abc'
    var_10 = {var_8: var_9}
    var_11 = 'STRING'
    var_12 = [var_11]
    var_13 = lambda self: var_12



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_eq_function_with_valid_number. Retrieved 11/17 statements.
# Partially parsed test_xpath_eq_function_with_second_index. Retrieved 11/17 statements.
# Partially parsed test_xpath_eq_function_with_non_number_raises_error. Retrieved 11/18 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'NUMBER'
    var_5 = [var_4]
    var_6 = 'arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'NUMBER'
    var_5 = [var_4]
    var_6 = 'arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '1'
    var_10 = {var_8: var_9}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'STRING'
    var_5 = [var_4]
    var_6 = 'arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'test'
    var_10 = {var_8: var_9}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_gt_function_raises_error_for_non_number_argument. Retrieved 14/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'text'
    var_13 = {var_11: var_12}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_lt_function_with_non_number_arguments. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'foo'
    var_13 = {var_11: var_12}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_contains_function_accepts_string_argument_type. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'STRING'
    var_4 = 'test'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_eq_function_predicate_true. Retrieved 12/24 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '0'
    var_11 = {var_9: var_10}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_lt_function_returns_xpath_with_position_less_than_one. Retrieved 14/23 statements.
# Partially parsed test_xpath_lt_function_raises_error_for_non_number. Retrieved 12/18 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'test'
    var_10 = [var_9]
    var_11 = {var_4: var_8, var_5: var_10}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_gt_function_non_number_argument_types_raises_expression_error. Retrieved 8/13 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = []
    var_4 = module_1.Function(var_2, var_3)
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = var_0.xpath_gt_function(var_1, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_eq_function_raises_error_for_non_number_argument. Retrieved 16/23 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpath_eq_function
    var_2 = 'eq'
    var_3 = 0
    var_4 = '"string"'
    var_5 = parse(var_4)[var_3]
    var_6 = [var_5]
    var_7 = module_1.Function(var_2, var_6)
    var_8 = False
    var_9 = 'xpath'
    var_10 = ()
    var_11 = 'add_post_condition'
    var_12 = None
    var_13 = lambda self, x: var_12
    var_14 = {var_11: var_13}
    var_15 = True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_eq_function_returns_correct_condition. Retrieved 14/21 statements.
# Partially parsed test_xpath_eq_function_with_index_1. Retrieved 14/21 statements.
# Partially parsed test_xpath_eq_function_raises_error_for_non_number. Retrieved 16/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'text'
    var_13 = {var_11: var_12}
    var_14 = False
    var_15 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_eq_function_with_non_number_argument. Retrieved 19/29 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = 'MockFunction'
    var_1 = ()
    var_2 = 'argument_types'
    var_3 = 'arguments'
    var_4 = 'STRING'
    var_5 = [var_4]
    var_6 = lambda self: var_5
    var_7 = 'MockArgument'
    var_8 = ()
    var_9 = 'value'
    var_10 = 'abc'
    var_11 = {var_9: var_10}
    var_12 = 'MockXPath'
    var_13 = ()
    var_14 = 'add_post_condition'
    var_15 = None
    var_16 = lambda self, cond: var_15
    var_17 = {var_14: var_16}
    var_18 = module_0.JQueryTranslator()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_has_function_non_string_non_ident_raises_expression_error. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 42



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_has_function_returns_elements_with_matching_descendant. Retrieved 4/7 statements.
# Partially parsed test_xpath_has_function_returns_empty_when_no_match. Retrieved 4/7 statements.
# Partially parsed test_xpath_has_function_works_with_tag_selector. Retrieved 3/6 statements.
# Partially parsed test_xpath_has_function_raises_error_on_invalid_argument_type. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '.bar'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '.baz'
    var_3 = [var_2]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = [var_1]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 123
    var_3 = [var_2]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_has_function_with_matching_selector. Retrieved 16/25 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Arg'
    var_12 = ()
    var_13 = 'value'
    var_14 = '.bar'
    var_15 = {var_13: var_14}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_lt_function_with_number_argument. Retrieved 4/9 statements.
# Partially parsed test_xpath_lt_function_with_negative_number. Retrieved 4/9 statements.
# Partially parsed test_xpath_lt_function_with_zero. Retrieved 4/9 statements.
# Partially parsed test_xpath_lt_function_raises_error_on_non_number. Retrieved 4/10 statements.
# Partially parsed test_xpath_lt_function_raises_error_on_multiple_arguments. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'lt'
    var_3 = '2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'lt'
    var_3 = '-1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'lt'
    var_3 = '0'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'lt'
    var_3 = 'abc'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'lt'
    var_3 = '1'
    var_4 = '2'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_contains_function_raises_expression_error_on_invalid_argument_types. Retrieved 3/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_lt_function_valid_number_argument. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'lt'
    var_4 = '1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 4/9 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 4/9 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 4/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = 1



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_gt_function_invalid_argument_type. Retrieved 4/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = var_0.xpath_gt_function(var_2, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_gt_function_raises_error_on_non_number. Retrieved 1/6 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error_for_non_number_argument. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'string'
    var_4 = [var_3]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_xpath_has_function_accepts_string. Retrieved 9/13 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'has'
    var_3 = 0
    var_4 = '"test"'
    var_5 = parse(var_4)[var_3]
    var_6 = var_5.parsed_tree
    var_7 = [var_6]
    var_8 = module_1.Function(var_2, var_7)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_on_non_number. Retrieved 4/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = 'STRING'
    var_3 = 'foo'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_xpath_lt_function_with_number_argument. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_with_non_number_argument_raises_error. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'test'
    var_13 = {var_11: var_12}



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_type. Retrieved 3/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'NUMBER'
    var_2 = module_0.JQueryTranslator()



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_gt_function_with_number_argument. Retrieved 14/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_lt_function_returns_correct_post_condition. Retrieved 14/22 statements.
# Partially parsed test_xpath_lt_function_with_zero_index. Retrieved 14/22 statements.
# Partially parsed test_xpath_lt_function_raises_error_for_non_number. Retrieved 14/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'text'
    var_13 = {var_11: var_12}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_disabled_pseudo. Retrieved 2/4 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//input'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_has_function_matches_selector. Retrieved 14/21 statements.
# Partially parsed test_xpath_has_function_no_match. Retrieved 14/21 statements.
# Partially parsed test_xpath_has_function_with_ident. Retrieved 14/21 statements.
# Partially parsed test_xpath_has_function_raises_on_invalid_args. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.bar'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.baz'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'div'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 2/3 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpathexpr_cls



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_contains_function_returns_xpath_with_contains_condition. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_raises_expression_error_for_non_string_or_ident. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_eq_function_returns_first_element. Retrieved 4/8 statements.
# Partially parsed test_xpath_eq_function_returns_second_element. Retrieved 4/8 statements.
# Partially parsed test_xpath_eq_function_raises_error_for_non_number. Retrieved 4/10 statements.
# Partially parsed test_xpath_eq_function_raises_error_for_multiple_arguments. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'eq'
    var_2 = '0'
    var_3 = module_0.JQueryTranslator()

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'eq'
    var_2 = '1'
    var_3 = module_0.JQueryTranslator()

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'eq'
    var_2 = 'text'
    var_3 = module_0.JQueryTranslator()

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'eq'
    var_2 = '0'
    var_3 = '1'
    var_4 = module_0.JQueryTranslator()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_has_function_invalid_argument_type_raises_error. Retrieved 13/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_lt_function_with_valid_number. Retrieved 4/10 statements.
# Partially parsed test_xpath_lt_function_with_zero_index. Retrieved 4/10 statements.
# Partially parsed test_xpath_lt_function_with_negative_number_raises_error. Retrieved 4/11 statements.
# Partially parsed test_xpath_lt_function_with_invalid_argument_type_raises_error. Retrieved 3/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'NUMBER'
    var_3 = '2'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'NUMBER'
    var_3 = '0'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'NUMBER'
    var_3 = '-1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'STRING'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'title'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_has_function_argument_types_is_STRING. Retrieved 8/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'string'
    var_3 = '"test"'
    var_4 = 0
    var_5 = (var_4, var_4)
    var_6 = 'has'
    var_7 = 'pseudo-class'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_eq_function_raises_on_non_number. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/test'
    var_2 = 'MockFunction'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'MockArg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'foo'
    var_13 = {var_11: var_12}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_gt_function_with_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1:gt(0)'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2[var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_has_function_valid_argument_type_string. Retrieved 6/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = '.bar'
    var_4 = 'descendant::*[contains(concat(" ", @class, " "), " bar ")]'
    var_5 = 'descendant::'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 12/21 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 12/21 statements.
# Partially parsed test_xpath_contains_function_raises_expression_error_for_invalid_argument_types. Retrieved 12/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = 'value'
    var_10 = 'title'
    var_11 = {var_9: var_10}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = 'value'
    var_10 = 'title'
    var_11 = {var_9: var_10}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = 'value'
    var_10 = '1'
    var_11 = {var_9: var_10}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 5/11 statements.
# Partially parsed test_xpath_contains_function_returns_updated_xpath. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"title"'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'title'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '42'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"text"'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_has_function_raises_expression_error_for_non_string_non_ident. Retrieved 8/15 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = []
    var_4 = module_1.Function(var_2, var_3)
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = var_0.xpath_has_function(var_1, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_lt_function_returns_xpath_with_correct_position_condition. Retrieved 12/21 statements.
# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 12/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '0'
    var_11 = {var_9: var_10}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = 'text'
    var_11 = {var_9: var_10}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_contains_function_valid_argument. Retrieved 6/8 statements.
# Partially parsed test_xpath_contains_function_valid_ident. Retrieved 6/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"test"'
    var_5 = [var_4]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'test'
    var_5 = [var_4]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_expression_error_raised_for_non_number_argument. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_1.object()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'text'
    var_13 = {var_11: var_12}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_gt_function_predicate_false. Retrieved 14/28 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = False
    var_13 = True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number. Retrieved 2/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_lt_function_with_non_number_raises_expression_error. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 'STRING'
    var_4 = 'text'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_lt_function_with_number_argument. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_with_non_number_argument. Retrieved 13/21 statements.
# Partially parsed test_xpath_lt_function_with_zero_index. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = {var_11: var_1}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_gt_function_valid_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = '0'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_eq_function_raises_error_on_non_number_argument. Retrieved 2/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_lt_function_with_non_number_argument_types. Retrieved 12/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'FakeFunction'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = 'test'
    var_11 = {var_9: var_10}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_eq_function_valid_number. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_non_string_non_ident_argument. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '123'
    var_13 = {var_11: var_12}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_lt_function_returns_xpath_with_position_condition. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 14/22 statements.
# Partially parsed test_xpath_lt_function_raises_error_for_empty_argument. Retrieved 10/15 statements.
# Partially parsed test_xpath_lt_function_with_zero_index. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'foo'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = []
    var_9 = {var_4: var_7, var_5: var_8}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_contains_function_valid_string_argument. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'MockFunction'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'value'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'MockArgument'
    var_11 = ()
    var_12 = 'test'
    var_13 = {var_6: var_12}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_gt_function_with_non_number_argument. Retrieved 13/24 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'Argument'
    var_4 = ()
    var_5 = 'value'
    var_6 = 'argument_types'
    var_7 = 'not_a_number'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda : var_9
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = [var_8]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_eq_function_raises_error_for_non_number_argument. Retrieved 7/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 'STRING'
    var_4 = 'foo'
    var_5 = False
    var_6 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_xpath_has_function_raises_expression_error_on_invalid_argument. Retrieved 5/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'NUMBER'
    var_4 = '42'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_non_string_or_ident_argument. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '123'
    var_13 = {var_11: var_12}



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_xpath_contains_function_raises_for_invalid_argument. Retrieved 8/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = []
    var_4 = 'NUMBER'
    var_5 = [var_4]
    var_6 = False
    var_7 = True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_xpath_contains_function_string. Retrieved 14/21 statements.
# Partially parsed test_xpath_contains_function_ident. Retrieved 14/21 statements.
# Partially parsed test_xpath_contains_function_raises_error. Retrieved 14/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'content'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/test'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_xpath_eq_function_with_valid_number. Retrieved 12/22 statements.
# Partially parsed test_xpath_eq_function_with_second_index. Retrieved 12/22 statements.
# Partially parsed test_xpath_eq_function_with_non_number_raises_error. Retrieved 11/22 statements.
# Partially parsed test_xpath_eq_function_preserves_existing_conditions. Retrieved 13/24 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '0'
    var_11 = {var_9: var_10}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '2'
    var_11 = {var_9: var_10}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = {var_9: var_1}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'existing_condition'
    var_3 = 'Function'
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_on_non_number_argument. Retrieved 11/20 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'obj'
    var_5 = 'value'
    var_6 = 'text'
    var_7 = {var_5: var_6}
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = var_0.xpath_eq_function(var_1, var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 14/21 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 14/21 statements.
# Partially parsed test_xpath_contains_function_with_invalid_argument_types. Retrieved 14/22 statements.
# Partially parsed test_xpath_contains_function_returns_xpath. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '5'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'test'
    var_13 = {var_11: var_12}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_xpath_lt_function_inside_xpath_translator. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_zero_index. Retrieved 14/21 statements.
# Partially parsed test_xpath_lt_function_large_index. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '10'
    var_13 = {var_11: var_12}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_xpath_has_function_returns_elements_with_matching_descendant. Retrieved 2/6 statements.
# Partially parsed test_xpath_has_function_returns_empty_when_no_matching_descendant. Retrieved 2/6 statements.
# Partially parsed test_xpath_has_function_returns_empty_when_self_matches_selector. Retrieved 2/6 statements.
# Partially parsed test_xpath_has_function_works_with_element_selector. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '<div class="foo"><div class="bar"></div></div>'
    var_1 = '.foo:has(".bar")'

def test_case_0():
    var_0 = '<div class="foo"><div class="bar"></div></div>'
    var_1 = '.foo:has(".baz")'

def test_case_0():
    var_0 = '<div class="foo"><div class="bar"></div></div>'
    var_1 = '.foo:has(".foo")'

def test_case_0():
    var_0 = '<div class="foo"><div class="bar"></div></div>'
    var_1 = '.foo:has(div)'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_xpath_has_function_raises_expression_error_for_non_string_non_ident_argument. Retrieved 14/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_xpath_eq_function_with_number_argument. Retrieved 11/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'obj'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_xpath_lt_function_with_non_number_argument_raises_expression_error. Retrieved 13/21 statements.


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = []
    var_4 = module_1.Function(var_2, var_3)
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = 'Arg'
    var_8 = ()
    var_9 = 'value'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = var_0.xpath_lt_function(var_1, var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_xpath_contains_function_with_valid_string_argument. Retrieved 1/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_xpath_eq_function_returns_correct_xpath_for_first_element. Retrieved 14/22 statements.
# Partially parsed test_xpath_eq_function_returns_correct_xpath_for_second_element. Retrieved 14/22 statements.
# Partially parsed test_xpath_eq_function_raises_error_for_non_number_argument. Retrieved 14/23 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '1'
    var_13 = {var_11: var_12}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'test'
    var_13 = {var_11: var_12}



