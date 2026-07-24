####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_eq_function_with_valid_index. Retrieved 19/21 statements.
# Partially parsed test_xpath_eq_function_with_invalid_index. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
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

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'invalid'
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
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() < 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() < 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 1/2 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpathexpr_cls



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_eq_function_non_number_argument. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'test'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.JQueryTranslator(var_0)
    var_2 = var_1.lower_case_attribute_names
    assert var_2 is False



# Parsed testcases at query #6
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.JQueryTranslator(var_0)
    var_2 = var_1.lower_case_attribute_names
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_contains_function_with_string. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_with_invalid_argument_type. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'test'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'test'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 123
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = None
    var_3 = 'invalid'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_has_function_with_valid_selector. Retrieved 19/22 statements.
# Partially parsed test_xpath_has_function_with_invalid_selector. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
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
    var_23 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '123'
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
    var_23 = bool(False)
    assert var_23 is True
    var_24 = 'Expected a single string or ident for :has(), got'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 6/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = '0'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_has_function_with_invalid_argument_types. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_contains_function. Retrieved 5/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'test'
    var_3 = '"test"'
    var_4 = 'contains(., "test")'



# Parsed testcases at query #13
#--------------------------




import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockFunction'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'MockArg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 123
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
    var_23 = None
    var_24 = var_0.xpath_contains_function(var_23, var_22)
    var_25 = bool(False)
    assert var_25 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//p'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = {}
    var_13 = module_1.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = [var_14]
    var_16 = 'NUMBER'
    var_17 = [var_16]
    var_18 = lambda : var_17
    var_19 = {var_4: var_15, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_arg. Retrieved 8/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = [var_3]
    var_5 = lambda : var_4
    var_6 = 'invalid'
    var_7 = [var_6]
    var_8 = "Expected a single integer for :gt(), got ['invalid']"



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'lt'
    var_4 = 'invalid'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'contains'
    var_4 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_contains_function_with_valid_string. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'test'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_eq_function_raises_error_for_non_number_argument. Retrieved 8/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = [var_3]
    var_5 = lambda : var_4
    var_6 = 'invalid'
    var_7 = [var_6]
    var_8 = "Expected a single integer for :eq(), got ['invalid']"



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 3/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'invalid'
    var_3 = "Expected a single string or ident for :contains(), got ['invalid']"



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_lt_function_non_number_argument. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'test'
    var_6 = [var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error_for_non_number_argument. Retrieved 7/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = "Expected a single integer for :gt(), got ['invalid']"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error. Retrieved 8/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = var_0.xpath_gt_function



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_eq_function_with_non_number_argument. Retrieved 4/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = 'test'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 8/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = var_0.xpath_eq_function



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_has_function_predicate. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'STRING'
    var_1 = [var_0]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() = 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() = 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 7/20 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() > 1'])
    assert var_5 is True
    var_6 = 1
    var_7 = var_1.post_conditions
    var_8 = bool(var_1.post_conditions == ['position() > 1', 'position() > 2'])
    assert var_8 is True
    var_9 = 'gt'
    var_10 = 'invalid'
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() < 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() < 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_has_function_with_invalid_argument_types. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '123'
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
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 7/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = "Expected a single integer for :eq(), got ['invalid']"



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_lt_function_with_invalid_argument_type. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 'invalid'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_contains_function. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/h1'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'contains'
    var_4 = 'title'
    var_5 = str(var_2)
    assert var_5 == './/h1[contains(., "title")]'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_lt_function_raises_expression_error_for_non_number_argument. Retrieved 7/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = "Expected a single integer for :gt(), got ['invalid']"



# Parsed testcases at query #35
#--------------------------




import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockFunction'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'MockArg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
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
    var_23 = None
    var_24 = var_0.xpath_has_function(var_23, var_22)
    assert var_24 is None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_has_function_predicate_true. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_xpath_has_function_with_matching_selector. Retrieved 19/22 statements.
# Partially parsed test_xpath_has_function_with_non_matching_selector. Retrieved 19/22 statements.
# Partially parsed test_xpath_has_function_with_tag_selector. Retrieved 19/22 statements.
# Partially parsed test_xpath_has_function_with_invalid_argument_type. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
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
    var_23 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.baz'
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
    var_23 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'div'
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
    var_23 = 'descendant::div'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '123'
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
    var_23 = bool(False)
    assert var_23 is True
    var_24 = 'Expected a single string or ident for :has(), got'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 5/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 0
    var_4 = str(var_1)
    assert var_4 == 'position() > 1'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_xpath_has_function_with_matching_selector. Retrieved 20/23 statements.
# Partially parsed test_xpath_has_function_with_non_matching_selector. Retrieved 20/23 statements.
# Partially parsed test_xpath_has_function_with_tag_selector. Retrieved 20/23 statements.
# Partially parsed test_xpath_has_function_with_invalid_argument_type. Retrieved 20/23 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/*'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = [var_17]
    var_19 = {var_4: var_8, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/*'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.baz'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = [var_17]
    var_19 = {var_4: var_8, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/*'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'div'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = [var_17]
    var_19 = {var_4: var_8, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/*'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '123'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = [var_17]
    var_19 = {var_4: var_8, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = bool(False)
    assert var_24 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_xpath_contains_function_with_invalid_argument_type. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'INVALID_TYPE'
    var_4 = 'invalid_arg'
    var_5 = bool(False)
    assert var_5 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 8/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = './/*'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'gt'
    var_4 = 0
    var_5 = str(var_2)
    assert var_5 == './/*[position() > 1]'
    var_6 = 1
    var_7 = str(var_2)
    assert var_7 == './/*[position() > 2]'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_eq_function_valid_index. Retrieved 19/21 statements.
# Partially parsed test_xpath_eq_function_invalid_argument_type. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
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

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'invalid'
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
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 1/2 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpathexpr_cls



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_has_function. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'div'
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['descendant::div'])
    assert var_5 is True



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

# Partially parsed test_xpath_lt_function_with_valid_index. Retrieved 17/19 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Argument'
    var_6 = ()
    var_7 = 'value'
    var_8 = '0'
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = 'NUMBER'
    var_15 = [var_14]
    var_16 = lambda : var_15
    var_17 = {var_3: var_13, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)



# Parsed testcases at query #7
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpathexpr_cls



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_gt_function_raises_error_for_non_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'gt'
    var_4 = 'invalid'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_has_function_with_invalid_argument_type. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() > 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() > 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_eq_function_with_non_number_argument. Retrieved 7/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 'invalid'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'eq'
    var_4 = 'invalid'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() < 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() < 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #15
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.lower_case_attribute_names
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_has_function_with_matching_selector. Retrieved 18/21 statements.
# Partially parsed test_xpath_has_function_with_non_matching_selector. Retrieved 18/21 statements.
# Partially parsed test_xpath_has_function_with_element_selector. Retrieved 18/21 statements.
# Partially parsed test_xpath_has_function_with_invalid_argument_type. Retrieved 18/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Argument'
    var_6 = ()
    var_7 = 'value'
    var_8 = '.bar'
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = 'STRING'
    var_15 = [var_14]
    var_16 = lambda : var_15
    var_17 = {var_3: var_13, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Argument'
    var_6 = ()
    var_7 = 'value'
    var_8 = '.baz'
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = 'STRING'
    var_15 = [var_14]
    var_16 = lambda : var_15
    var_17 = {var_3: var_13, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " baz ")]'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Argument'
    var_6 = ()
    var_7 = 'value'
    var_8 = 'div'
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = 'STRING'
    var_15 = [var_14]
    var_16 = lambda : var_15
    var_17 = {var_3: var_13, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'descendant::div'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Argument'
    var_6 = ()
    var_7 = 'value'
    var_8 = 123
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = 'NUMBER'
    var_15 = [var_14]
    var_16 = lambda : var_15
    var_17 = {var_3: var_13, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = bool(False)
    assert var_22 is True
    var_23 = 'Expected a single string or ident for :has()'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_gt_function_with_non_number_argument. Retrieved 3/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'invalid'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_contains_function_with_string. Retrieved 19/21 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 19/21 statements.
# Partially parsed test_xpath_contains_function_with_invalid_arg. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
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

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
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

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 123
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
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'test'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_gt_function_with_valid_integer. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = '0'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_invalid_argument_types. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
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
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_contains_function. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 5/10 statements.
# Partially parsed test_xpath_contains_function_invalid_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'contains'
    var_4 = 'test'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'contains'
    var_4 = 'test'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'contains'
    var_4 = 123
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_has_function. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'div'
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['descendant::div'])
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 6/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() < 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() < 2'])
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 8/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() = 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() = 2'])
    assert var_9 is True
    var_10 = module_0.XPathExpr()
    var_11 = 'invalid'
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = '123'
    var_6 = [var_5]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'invalid'
    var_5 = [var_4]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_contains_function_raises_error_for_invalid_arguments. Retrieved 8/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = [var_3]
    var_5 = lambda : var_4
    var_6 = '123'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 6/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'invalid'
    var_5 = [var_4]
    var_6 = "Expected a single integer for :eq(), got ['invalid']"



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_eq_function_raises_expression_error_for_non_number_argument. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = 'invalid'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_lt_function_raises_expression_error_for_non_number_argument. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_has_function. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'has'
    var_4 = '".bar"'
    var_5 = str(var_2)
    assert var_5 == '//div[descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]]'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_has_function_with_invalid_argument_type. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_gt_function_raises_error_for_non_number_argument. Retrieved 7/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_gt_function_with_non_number_argument. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = '0'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_xpath_contains_function_raises_expression_error_for_invalid_argument_types. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ''
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'INVALID'
    var_4 = 'invalid_arg'
    var_5 = "Expected a single string or ident for :contains(), got ['invalid_arg']"



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 6/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 0
    var_4 = var_1.post_conditions
    var_5 = bool(var_1.post_conditions == ['position() = 1'])
    assert var_5 is True
    var_6 = module_0.XPathExpr()
    var_7 = 1
    var_8 = var_6.post_conditions
    var_9 = bool(var_6.post_conditions == ['position() = 2'])
    assert var_9 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error_for_non_number_argument. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'invalid'
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_xpath_contains_function. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'test'



