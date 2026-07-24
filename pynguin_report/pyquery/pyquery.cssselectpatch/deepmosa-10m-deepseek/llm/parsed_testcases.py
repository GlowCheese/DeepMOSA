####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 20/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_hidden_pseudo. Retrieved 6/20 statements.


def test_case_0():
    var_0 = '<div><input type="hidden"/></div>'
    var_1 = 'input:hidden'
    var_2 = 0
    var_3 = 'type'
    var_4 = '<div><input type="text"/></div>'
    var_5 = '<div><input type="hidden"/><input type="text"/></div>'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_xpath_even_pseudo. Retrieved 2/4 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_xpath_eq_function_with_number_argument. Retrieved 13/20 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = '0'
    var_4 = [var_3]
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = 'Arg'
    var_8 = ()
    var_9 = 'value'
    var_10 = {var_9: var_3}
    var_11 = [var_7, var_8, var_10]
    var_12 = {}
    var_13 = module_1.type(*var_11, **var_12)
    var_14 = var_13()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_eq_function_with_non_number_argument_types. Retrieved 5/12 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = {}
    var_3 = module_1.object(*var_1, **var_2)
    var_4 = 'eq'
    var_5 = 'IDENT'
    var_6 = 'foo'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_eq_function_predicate_false. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 'not_a_number'
    var_4 = (var_3,)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = '0'
    var_3 = 0
    var_4 = 'h1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_accepts_number_argument_type. Retrieved 3/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 19/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 1/2 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = var_0.xpathexpr_cls
    var_3 = bool(var_0.xpathexpr_cls is not None)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_lt_function_returns_xpath_with_position_less_than_positive_index. Retrieved 4/8 statements.
# Partially parsed test_xpath_lt_function_returns_xpath_with_position_less_than_zero_index. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = '1'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = '0'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_lt_function_with_non_number_argument_returns_false. Retrieved 3/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_contains_function_returns_xpath_with_contains_condition. Retrieved 20/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = 'some text'
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_has_function_matching_selector. Retrieved 3/7 statements.
# Partially parsed test_xpath_has_function_no_matching_selector. Retrieved 3/7 statements.
# Partially parsed test_xpath_has_function_with_ident. Retrieved 3/7 statements.
# Partially parsed test_xpath_has_function_raises_on_invalid_argument_type. Retrieved 3/7 statements.
# Partially parsed test_xpath_has_function_returns_xpath. Retrieved 3/6 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = '.bar'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.baz'
    var_3 = '.baz'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'IDENT'
    var_2 = 'div'
    var_3 = 'div'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '1'
    var_3 = bool(False)
    assert var_3 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.test'



# Parsed testcases at query #15
#--------------------------




import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_lt_function_incorrect_argument_types. Retrieved 12/22 statements.


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
    var_10 = '0'
    var_11 = {var_9: var_10}
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------




import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = 'a'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = var_0.xpath_gt_function(var_1, var_22)
    var_24 = bool(False)
    assert var_24 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_has_function_string_argument. Retrieved 6/8 statements.
# Partially parsed test_xpath_has_function_ident_argument. Retrieved 6/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = '"div"'
    var_4 = [var_2, var_3]
    var_5 = module_0.XPathExpr()

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = 'div'
    var_4 = [var_2, var_3]
    var_5 = module_0.XPathExpr()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_has_function_raises_error_for_invalid_argument_type. Retrieved 2/12 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line18_returns_false_for_valid_argument_types. Retrieved 7/17 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = 'has'
    var_3 = 'string'
    var_4 = '"bar"'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_gt_function_accepts_number_argument. Retrieved 7/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = 0
    var_3 = '0'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = 'h1'
    var_7 = 'position() > 1'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 12/21 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 12/21 statements.
# Partially parsed test_xpath_contains_function_with_invalid_argument_type. Retrieved 12/22 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = 'title'
    var_11 = {var_9: var_10}
    var_12 = 'contains(., "title")'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = 'title'
    var_11 = {var_9: var_10}
    var_12 = 'contains(., "title")'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'Function'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '1'
    var_11 = {var_9: var_10}
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_contains_function_returns_correct_xpath. Retrieved 22/24 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 22/24 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 22/25 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpath_cls.path
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = '__init__'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = 'title'
    var_14 = {var_12: var_13}
    var_15 = [var_10, var_11, var_14]
    var_16 = {}
    var_17 = module_1.type(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = None
    var_20 = lambda self: var_19
    var_21 = {var_4: var_9, var_5: var_18, var_6: var_20}
    var_22 = [var_2, var_3, var_21]
    var_23 = {}
    var_24 = module_1.type(*var_22, **var_23)
    var_25 = var_24()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpath_cls.path
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = '__init__'
    var_7 = 'IDENT'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = 'text'
    var_14 = {var_12: var_13}
    var_15 = [var_10, var_11, var_14]
    var_16 = {}
    var_17 = module_1.type(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = None
    var_20 = lambda self: var_19
    var_21 = {var_4: var_9, var_5: var_18, var_6: var_20}
    var_22 = [var_2, var_3, var_21]
    var_23 = {}
    var_24 = module_1.type(*var_22, **var_23)
    var_25 = var_24()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpath_cls.path
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = '__init__'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '1'
    var_14 = {var_12: var_13}
    var_15 = [var_10, var_11, var_14]
    var_16 = {}
    var_17 = module_1.type(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = None
    var_20 = lambda self: var_19
    var_21 = {var_4: var_9, var_5: var_18, var_6: var_20}
    var_22 = [var_2, var_3, var_21]
    var_23 = {}
    var_24 = module_1.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = bool(False)
    assert var_26 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_eq_function_with_number. Retrieved 20/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_xpath_has_function_predicate_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '<div class="foo"><div class="bar"></div></div>'
    var_1 = '.foo:has(".bar")'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_contains_function_returns_xpath_with_contains_condition. Retrieved 11/20 statements.
# Partially parsed test_xpath_contains_function_returns_xpath_with_contains_condition_ident. Retrieved 11/20 statements.
# Partially parsed test_xpath_contains_function_returns_same_xpath_instance. Retrieved 11/20 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_types. Retrieved 11/21 statements.


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
    var_10 = {var_9: var_1}

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
    var_10 = {var_9: var_1}

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
    var_10 = {var_9: var_1}

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
    var_10 = {var_9: var_1}
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_eq_function_raises_error_for_non_number_argument. Retrieved 4/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'eq'
    var_2 = ()
    var_3 = module_0.XPathExpr()
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_xpath_gt_function_raises_error_when_argument_types_not_number. Retrieved 5/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = 'IDENT'
    var_3 = 'foo'
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_lt_function_with_number_argument. Retrieved 2/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'position() < 3'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_has_function_accepts_string_argument. Retrieved 5/15 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = "descendant::*[contains(concat(' ', @class, ' '), ' bar ')]"
    var_4 = 'descendant::'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_lt_function_basic. Retrieved 20/22 statements.
# Partially parsed test_xpath_lt_function_negative_index. Retrieved 20/22 statements.
# Partially parsed test_xpath_lt_function_large_index. Retrieved 20/22 statements.
# Partially parsed test_xpath_lt_function_returns_xpath. Retrieved 20/23 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '1'
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
    var_24 = 'position() < 2'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_24 = 'position() < 1'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
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
    var_24 = 'position() < 11'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'span'
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_lt_function. Retrieved 2/3 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_lt_function_with_valid_number. Retrieved 12/21 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'FakeFunction'
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = 'value'
    var_10 = '0'
    var_11 = {var_9: var_10}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_contains_function_predicate_false. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 'NUMBER'
    var_3 = '123'
    var_4 = 'div'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_has_function_valid_argument_types. Retrieved 6/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'has'
    var_2 = 'STRING'
    var_3 = '.bar'
    var_4 = (var_2, var_3)
    var_5 = [var_4]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_xpath_image_pseudo_adds_condition. Retrieved 2/5 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = "@type = 'image' and name(.) = 'input'"



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 3/8 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = 'position() > 1'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_jquery_translator_constructor. Retrieved 1/2 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = var_0.xpathexpr_cls



# Parsed testcases at query #4
#--------------------------




import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '2'
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
    var_24 = var_0.xpath_lt_function(var_1, var_23)
    var_25 = var_24.post_conditions
    var_26 = bool(var_24.post_conditions == ['position() < 3'])
    assert var_26 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
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
    var_12 = 'text'
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
    var_24 = var_0.xpath_lt_function(var_1, var_23)
    var_25 = bool(False)
    assert var_25 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '0'
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
    var_24 = var_0.xpath_lt_function(var_1, var_23)
    var_25 = bool(var_24 is var_1)
    assert var_25 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_xpath_has_function_with_string_selector. Retrieved 5/7 statements.
# Partially parsed test_xpath_has_function_with_ident_selector. Retrieved 5/7 statements.
# Partially parsed test_xpath_has_function_returns_modified_xpath. Retrieved 5/7 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '"test"'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = [var_2]
    var_4 = 'test'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'div'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_xpath_lt_function_raises_error_for_non_number_argument. Retrieved 5/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 'dummy'
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_xpath_eq_function_with_number_argument. Retrieved 20/22 statements.
# Partially parsed test_xpath_eq_function_with_negative_number. Retrieved 20/22 statements.
# Partially parsed test_xpath_eq_function_with_non_number_argument_raises_error. Retrieved 20/23 statements.
# Partially parsed test_xpath_eq_function_with_multiple_arguments_raises_error. Retrieved 25/28 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
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
    var_18 = lambda self: var_17
    var_19 = {var_4: var_15, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = '-1'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = {}
    var_13 = module_1.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = [var_14]
    var_16 = 'NUMBER'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = {var_4: var_15, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = {}
    var_13 = module_1.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = [var_14]
    var_16 = 'STRING'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = {var_4: var_15, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = bool(False)
    assert var_24 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
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
    var_15 = ()
    var_16 = '1'
    var_17 = {var_8: var_16}
    var_18 = [var_6, var_15, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = [var_14, var_21]
    var_23 = 'NUMBER'
    var_24 = [var_23, var_23]
    var_25 = lambda self: var_24
    var_26 = {var_4: var_22, var_5: var_25}
    var_27 = [var_2, var_3, var_26]
    var_28 = {}
    var_29 = module_1.type(*var_27, **var_28)
    var_30 = var_29()
    var_31 = bool(False)
    assert var_31 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_xpath_eq_function_with_number_argument. Retrieved 7/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 'NUMBER'
    var_4 = '0'
    var_5 = (var_3, var_4)
    var_6 = [var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 20/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_xpath_eq_function_non_number_argument_raises_error. Retrieved 12/22 statements.


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
    var_10 = 'abc'
    var_11 = {var_9: var_10}
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_xpath_has_function_with_string_argument. Retrieved 5/7 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = '".baz"'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_xpath_contains_function_with_string_argument. Retrieved 5/7 statements.
# Partially parsed test_xpath_contains_function_with_ident_argument. Retrieved 5/7 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_invalid_argument_type. Retrieved 5/8 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_multiple_arguments. Retrieved 6/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = '"title"'
    var_4 = [var_3]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'IDENT'
    var_3 = 'title'
    var_4 = [var_3]

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = '1'
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'STRING'
    var_3 = '"a"'
    var_4 = '"b"'
    var_5 = [var_3, var_4]
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error_for_non_number_argument. Retrieved 5/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'string'
    var_4 = (var_3,)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_xpath_eq_function_with_non_number_argument_types_raises_expression_error. Retrieved 4/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = 'test'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_xpath_gt_function_with_non_number_argument. Retrieved 6/9 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.XPathExpr()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = '"test"'
    var_4 = [var_3]
    var_5 = module_0.JQueryTranslator()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_xpath_lt_function_raises_expression_error_for_non_number_argument. Retrieved 7/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'lt'
    var_3 = 'string'
    var_4 = 'abc'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_xpath_has_function_predicate_true_for_string_argument_type. Retrieved 6/13 statements.
# Partially parsed test_xpath_has_function_predicate_true_for_ident_argument_type. Retrieved 6/13 statements.
# Partially parsed test_xpath_has_function_predicate_false_for_number_argument_type. Retrieved 6/15 statements.
# Partially parsed test_xpath_has_function_predicate_false_for_empty_argument_types. Retrieved 6/15 statements.
# Partially parsed test_xpath_has_function_predicate_false_for_multiple_argument_types. Retrieved 7/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'STRING'
    var_4 = '"test"'
    var_5 = [var_3]
    var_6 = bool(True)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'IDENT'
    var_4 = 'test'
    var_5 = [var_3]
    var_6 = bool(True)
    assert var_6 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'NUMBER'
    var_4 = '123'
    var_5 = [var_3]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'STRING'
    var_4 = '"test"'
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'STRING'
    var_4 = '"test"'
    var_5 = 'IDENT'
    var_6 = [var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_xpath_contains_function_returns_none_when_argument_types_is_list_with_string. Retrieved 8/16 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'string'
    var_4 = '"test"'
    var_5 = 0
    var_6 = 'STRING'
    var_7 = [var_6]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_xpath_has_function_accepts_string_argument_type. Retrieved 20/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/'
    var_2 = 'MockFunction'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'MockArgument'
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_xpath_gt_function_predicate_false. Retrieved 7/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'not_a_number'
    var_4 = (var_3,)
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_xpath_eq_function_validates_argument_types. Retrieved 7/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = '0'
    var_4 = (var_3,)
    var_5 = 'NUMBER'
    var_6 = [var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_xpath_has_function_with_string_argument. Retrieved 20/22 statements.
# Partially parsed test_xpath_has_function_with_ident_argument. Retrieved 20/22 statements.
# Partially parsed test_xpath_has_function_with_invalid_arguments. Retrieved 19/22 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = "descendant::*[contains(concat(' ', normalize-space(@class), ' '), ' bar ')]"

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = 'descendant::div'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/html/body'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = bool(False)
    assert var_23 is True
    var_24 = 'Expected a single string or ident for :has()'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_xpath_has_function_returns_xpath_with_post_condition_when_argument_is_string. Retrieved 19/21 statements.
# Partially parsed test_xpath_has_function_returns_xpath_with_post_condition_when_argument_is_ident. Retrieved 19/21 statements.
# Partially parsed test_xpath_has_function_raises_expression_error_for_invalid_argument_types. Retrieved 19/22 statements.
# Partially parsed test_xpath_has_function_adds_correct_post_condition_for_string_argument. Retrieved 20/25 statements.
# Partially parsed test_xpath_has_function_adds_correct_post_condition_for_ident_argument. Retrieved 20/25 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = bool(False)
    assert var_23 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = 'descendant::'

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = 'descendant::'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_xpath_gt_function_raises_expression_error_for_non_number_arguments. Retrieved 7/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'not_a_number'
    var_4 = [var_3]
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------




import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'type'
    var_10 = 'title'
    var_11 = 'STRING'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = [var_6, var_7, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = [var_11]
    var_18 = lambda self: var_17
    var_19 = {var_4: var_16, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = var_0.xpath_contains_function(var_1, var_23)
    var_25 = var_24.post_conditions
    var_26 = bool(var_24.post_conditions == ["contains(., 'title')"])
    assert var_26 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'type'
    var_10 = 'title'
    var_11 = 'IDENT'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = [var_6, var_7, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = [var_11]
    var_18 = lambda self: var_17
    var_19 = {var_4: var_16, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = var_0.xpath_contains_function(var_1, var_23)
    var_25 = var_24.post_conditions
    var_26 = bool(var_24.post_conditions == ["contains(., 'title')"])
    assert var_26 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Arg'
    var_7 = ()
    var_8 = 'value'
    var_9 = 'type'
    var_10 = '1'
    var_11 = 'NUMBER'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = [var_6, var_7, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = [var_11]
    var_18 = lambda self: var_17
    var_19 = {var_4: var_16, var_5: var_18}
    var_20 = [var_2, var_3, var_19]
    var_21 = {}
    var_22 = module_1.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = var_0.xpath_contains_function(var_1, var_23)
    var_25 = bool(False)
    assert var_25 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_xpath_contains_function_with_string. Retrieved 18/20 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 18/20 statements.
# Partially parsed test_xpath_contains_function_raises_on_invalid_argument_type. Retrieved 18/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = {var_3: var_7, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = {var_3: var_7, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = {}
    var_15 = module_1.type(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = {var_3: var_7, var_4: var_16}
    var_18 = [var_1, var_2, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = bool(False)
    assert var_22 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_xpath_lt_function_with_number_argument. Retrieved 19/21 statements.
# Partially parsed test_xpath_lt_function_with_zero_argument. Retrieved 19/21 statements.
# Partially parsed test_xpath_lt_function_with_non_number_raises_error. Retrieved 19/22 statements.
# Partially parsed test_xpath_lt_function_with_negative_number. Retrieved 19/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = bool(False)
    assert var_23 is True

import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '-1'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_gt_function_with_non_number_argument. Retrieved 9/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'STRING'
    var_4 = 'foo'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = False
    var_8 = True
    var_9 = bool(var_8)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_xpath_gt_function_argument_types_returns_number. Retrieved 7/11 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = 'NUMBER'
    var_4 = '0'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_xpath_eq_function. Retrieved 31/35 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = module_0.JQueryTranslator()
    var_24 = ()
    var_25 = [var_6]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '2'
    var_29 = {var_11: var_28}
    var_30 = [var_9, var_27, var_29]
    var_31 = {}
    var_32 = module_1.type(*var_30, **var_31)
    var_33 = [var_32]
    var_34 = {var_4: var_26, var_5: var_33}
    var_35 = [var_2, var_24, var_34]
    var_36 = {}
    var_37 = module_1.type(*var_35, **var_36)
    var_38 = var_37()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_xpath_eq_function_accepts_number_argument_type. Retrieved 5/7 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = [var_2]
    var_4 = lambda : var_3



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_xpath_lt_function_raises_expression_error_for_non_number_argument. Retrieved 7/14 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'lt'
    var_3 = 0
    var_4 = 'string'
    var_5 = parse(var_4)[var_3]
    var_6 = (var_5,)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_xpath_gt_function. Retrieved 19/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_xpath_contains_function_invalid_argument_type_raises_error. Retrieved 5/10 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 'NUMBER'
    var_4 = '123'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_xpath_lt_function_argument_types_is_number. Retrieved 29/36 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = 'arg'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = ()
    var_24 = [var_6]
    var_25 = lambda : var_24
    var_26 = ()
    var_27 = {var_11: var_12}
    var_28 = [var_9, var_26, var_27]
    var_29 = {}
    var_30 = module_1.type(*var_28, **var_29)
    var_31 = [var_30]
    var_32 = {var_4: var_25, var_5: var_31}
    var_33 = [var_2, var_23, var_32]
    var_34 = {}
    var_35 = module_1.type(*var_33, **var_34)
    var_36 = var_35()
    var_37 = bool(True)
    assert var_37 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_xpath_lt_function_returns_correct_position_condition. Retrieved 19/21 statements.
# Partially parsed test_xpath_lt_function_with_zero_index. Retrieved 19/21 statements.
# Partially parsed test_xpath_lt_function_with_negative_number. Retrieved 19/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
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
    var_12 = '2'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()

import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
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
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #37
#--------------------------




import builtins as module_0
import pyquery.cssselectpatch as module_1

def test_case_0():
    var_0 = 'Function'
    var_1 = ()
    var_2 = 'argument_types'
    var_3 = 'arguments'
    var_4 = 'STRING'
    var_5 = [var_4]
    var_6 = lambda self: var_5
    var_7 = 'Argument'
    var_8 = ()
    var_9 = 'value'
    var_10 = 'hello'
    var_11 = {var_9: var_10}
    var_12 = [var_7, var_8, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = [var_15]
    var_17 = {var_2: var_6, var_3: var_16}
    var_18 = [var_0, var_1, var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'XPath'
    var_23 = ()
    var_24 = 'add_post_condition'
    var_25 = None
    var_26 = lambda self, cond: var_25
    var_27 = {var_24: var_26}
    var_28 = [var_22, var_23, var_27]
    var_29 = {}
    var_30 = module_0.type(*var_28, **var_29)
    var_31 = var_30()
    var_32 = module_1.JQueryTranslator()
    var_33 = var_32.xpath_lt_function(var_31, var_21)
    var_34 = bool(False)
    assert var_34 is True
    var_35 = bool(True)
    assert var_35 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_has_function_raises_for_invalid_arguments. Retrieved 7/13 statements.


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'has'
    var_3 = 'NUMBER'
    var_4 = '42'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_xpath_has_function_returns_xpath_with_post_condition. Retrieved 19/21 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {var_4: var_8, var_5: var_17}
    var_19 = [var_2, var_3, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_xpath_contains_function_with_string. Retrieved 20/22 statements.
# Partially parsed test_xpath_contains_function_with_ident. Retrieved 20/22 statements.
# Partially parsed test_xpath_contains_function_raises_error_for_non_string_or_ident. Retrieved 20/23 statements.


import pyquery.cssselectpatch as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/'
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
    var_1 = '/'
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
    var_12 = 'text'
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
    var_1 = '/'
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



