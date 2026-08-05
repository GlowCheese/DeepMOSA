####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


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
    var_8 = lambda self: var_7
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = 'div'
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'content'
    var_21 = {var_11: var_20}
    var_22 = 'Function'
    var_23 = ()
    var_24 = 'argument_types'
    var_25 = 'arguments'
    var_26 = 'NUMBER'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = 'Argument'
    var_30 = ()
    var_31 = 'value'
    var_32 = '123'
    var_33 = {var_31: var_32}



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = 'title'
    var_8 = parse(var_7)[var_2]
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = '123'
    var_12 = parse(var_11)[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = var_0.xpath_contains_function(var_1, var_14)



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = var_0.xpath_hidden_pseudo(var_1)
    var_3 = str(var_2)



# Parsed testcases at query #4
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = [var_6]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '0'
    var_20 = {var_11: var_19}
    var_21 = module_0.XPathExpr()
    var_22 = ()
    var_23 = [var_6]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = '-1'
    var_27 = {var_11: var_26}
    var_28 = module_0.XPathExpr()
    var_29 = ()
    var_30 = 'STRING'
    var_31 = [var_30]
    var_32 = lambda self: var_31
    var_33 = ()
    var_34 = 'test'
    var_35 = {var_11: var_34}



# Parsed testcases at query #5
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'
    var_4 = 'test'
    var_5 = '2'



# Parsed testcases at query #6
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Arg'
    var_28 = ()
    var_29 = 'value'
    var_30 = '1'
    var_31 = {var_29: var_30}
    var_32 = ()
    var_33 = [var_24]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = '.test'
    var_37 = {var_29: var_36}



# Parsed testcases at query #7
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'
    var_4 = '-1'
    var_5 = 'test'



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
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
    var_16 = module_0.XPathExpr(var_1, var_2)
    var_17 = ()
    var_18 = 'IDENT'
    var_19 = [var_18]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = {var_13: var_2}
    var_23 = module_0.XPathExpr(var_1, var_2)
    var_24 = ()
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '5'
    var_30 = {var_13: var_29}
    var_31 = module_0.XPathExpr(var_1, var_2)
    var_32 = ()
    var_33 = []
    var_34 = lambda self: var_33
    var_35 = []
    var_36 = {var_6: var_34, var_7: var_35}
    var_37 = module_0.XPathExpr(var_1, var_2)
    var_38 = ()
    var_39 = [var_8]
    var_40 = lambda self: var_39
    var_41 = ()
    var_42 = '.my-class'
    var_43 = {var_13: var_42}



# Parsed testcases at query #9
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(element=var_1)
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '.bar'
    var_14 = {var_12: var_13}
    var_15 = module_0.XPathExpr(element=var_1)
    var_16 = ()
    var_17 = 'IDENT'
    var_18 = [var_17]
    var_19 = lambda self: var_18
    var_20 = ()
    var_21 = {var_12: var_1}
    var_22 = "@class='foo'"
    var_23 = module_0.XPathExpr(element=var_1, condition=var_22)
    var_24 = ()
    var_25 = [var_7]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '.baz'
    var_29 = {var_12: var_28}
    var_30 = 'div'
    var_31 = module_0.XPathExpr(element=var_30)
    var_32 = 'Function'
    var_33 = ()
    var_34 = 'argument_types'
    var_35 = 'arguments'
    var_36 = 'NUMBER'
    var_37 = [var_36]
    var_38 = lambda self: var_37
    var_39 = 'Arg'
    var_40 = ()
    var_41 = 'value'
    var_42 = '0'
    var_43 = {var_41: var_42}



# Parsed testcases at query #10
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = {}
    var_4 = 'Arg'
    var_5 = ()
    var_6 = 'value'
    var_7 = 'title'
    var_8 = {var_6: var_7}
    var_9 = 'STRING'
    var_10 = [var_9]
    var_11 = ()
    var_12 = {}
    var_13 = ()
    var_14 = 'text'
    var_15 = {var_6: var_14}
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = ()
    var_19 = {}
    var_20 = ()
    var_21 = '1'
    var_22 = {var_6: var_21}
    var_23 = 'NUMBER'
    var_24 = [var_23]
    var_25 = ()
    var_26 = {}
    var_27 = ()
    var_28 = 'a'
    var_29 = {var_6: var_28}
    var_30 = ()
    var_31 = 'b'
    var_32 = {var_6: var_31}
    var_33 = [var_9, var_9]
    var_34 = ()
    var_35 = {}
    var_36 = ()
    var_37 = "it's"
    var_38 = {var_6: var_37}
    var_39 = [var_9]



# Parsed testcases at query #11
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'has'
    var_2 = '".bar"'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = module_1.Function(var_1, var_4)
    var_6 = 'div'
    var_7 = module_1.parse(var_6)
    var_8 = [var_7]
    var_9 = module_1.Function(var_1, var_8)
    var_10 = '123'
    var_11 = module_1.parse(var_10)
    var_12 = [var_11]
    var_13 = module_1.Function(var_1, var_12)
    var_14 = 'div'
    var_15 = var_0.xpath_has_function(var_2, var_13)



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()
    var_4 = []
    var_5 = module_0.XPathExpr()



# Parsed testcases at query #13
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = '.bar'
    var_5 = module_0.XPathExpr(var_1, var_2)
    var_6 = 'span'
    var_7 = module_0.XPathExpr(var_1, var_2)
    var_8 = '.foo'
    var_9 = module_0.XPathExpr(var_1, var_2)
    var_10 = '.test'
    var_11 = module_0.XPathExpr(var_1, var_2)
    var_12 = 'NUMBER'
    var_13 = var_0.xpath_has_function(var_11, var_2)



# Parsed testcases at query #14
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = ":contains('test')"
    var_3 = parse(var_2)[var_1]
    var_4 = var_3.parsed_selectors[var_1]
    var_5 = var_4.pseudo_class
    var_6 = ':contains(test)'
    var_7 = parse(var_6)[var_1]
    var_8 = var_7.parsed_selectors[var_1]
    var_9 = var_8.pseudo_class
    var_10 = ':contains(123)'
    var_11 = parse(var_10)[var_1]
    var_12 = var_11.parsed_selectors[var_1]
    var_13 = var_12.pseudo_class



# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'STRING'



# Parsed testcases at query #16
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #17
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'test'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '123'
    var_26 = {var_10: var_25}



# Parsed testcases at query #18
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'arguments'
    var_4 = 'argument_types'
    var_5 = 'Arg'
    var_6 = ()
    var_7 = 'value'
    var_8 = '0'
    var_9 = {var_7: var_8}
    var_10 = 'NUMBER'
    var_11 = [var_10]
    var_12 = lambda self: var_11
    var_13 = ()
    var_14 = ()
    var_15 = '1'
    var_16 = {var_7: var_15}
    var_17 = [var_10]
    var_18 = lambda self: var_17
    var_19 = 'Function'
    var_20 = ()
    var_21 = 'arguments'
    var_22 = 'argument_types'
    var_23 = 'Arg'
    var_24 = ()
    var_25 = 'value'
    var_26 = 'not_a_number'
    var_27 = {var_25: var_26}
    var_28 = 'STRING'
    var_29 = [var_28]
    var_30 = lambda self: var_29



# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'position() = 1'
    var_2 = 'condition1'



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = '0'
    var_4 = module_0.XPathExpr()
    var_5 = '1'
    var_6 = module_0.XPathExpr()
    var_7 = '-1'
    var_8 = module_0.XPathExpr()
    var_9 = 'STRING'
    var_10 = module_0.XPathExpr()
    var_11 = '2'



# Parsed testcases at query #21
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []
    var_2 = '0'
    var_3 = '1'
    var_4 = '5'



# Parsed testcases at query #22
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'content'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '1'
    var_26 = {var_10: var_25}



# Parsed testcases at query #23
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = [var_6]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = '-1'
    var_19 = {var_11: var_18}
    var_20 = 'h1'
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'STRING'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Argument'
    var_29 = ()
    var_30 = 'value'
    var_31 = 'test'
    var_32 = {var_30: var_31}



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = module_0.XPathExpr()
    var_4 = '2'
    var_5 = module_0.XPathExpr()
    var_6 = '-1'
    var_7 = module_0.XPathExpr()
    var_8 = 'string'
    var_9 = module_0.XPathExpr()
    var_10 = '1, 2'



# Parsed testcases at query #25
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '3'
    var_4 = 'STRING'
    var_5 = 'invalid'



# Parsed testcases at query #26
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '3'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #27
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = '3'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda : var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda : var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda : var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #28
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'div'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = 1
    var_29 = {var_11: var_28}



# Parsed testcases at query #29
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = '0'
    var_5 = module_0.XPathExpr(var_1, var_2)
    var_6 = '1'
    var_7 = module_0.XPathExpr(var_1, var_2)
    var_8 = '5'
    var_9 = module_0.XPathExpr(var_1, var_2)
    var_10 = 'test'
    var_11 = module_0.XPathExpr(var_1, var_2)
    var_12 = '2'
    var_13 = module_0.XPathExpr(var_1, var_2)



# Parsed testcases at query #30
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = 1
    var_3 = 5



# Parsed testcases at query #31
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #32
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '.bar'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = [var_7]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '.nonexistent'
    var_20 = {var_12: var_19}
    var_21 = ()
    var_22 = [var_7]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = 'span'
    var_26 = {var_12: var_25}
    var_27 = 'Function'
    var_28 = ()
    var_29 = 'argument_types'
    var_30 = 'arguments'
    var_31 = 'NUMBER'
    var_32 = [var_31]
    var_33 = lambda self: var_32
    var_34 = 'Arg'
    var_35 = ()
    var_36 = 'value'
    var_37 = '1'
    var_38 = {var_36: var_37}
    var_39 = [var_14]
    var_40 = {var_29: var_33, var_30: var_39}



# Parsed testcases at query #33
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = '0'
    var_4 = '-1'
    var_5 = 'STRING'



# Parsed testcases at query #34
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = '1'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = module_1.Function(var_1, var_4)
    var_6 = '0'
    var_7 = module_1.parse(var_6)
    var_8 = [var_7]
    var_9 = module_1.Function(var_1, var_8)
    var_10 = '-1'
    var_11 = module_1.parse(var_10)
    var_12 = [var_11]
    var_13 = module_1.Function(var_1, var_12)
    var_14 = 'gt'
    var_15 = '"string"'
    var_16 = module_1.parse(var_15)
    var_17 = [var_16]
    var_18 = module_1.Function(var_14, var_17)
    var_19 = 'gt'
    var_20 = '1'
    var_21 = module_1.parse(var_20)
    var_22 = '2'
    var_23 = module_1.parse(var_22)
    var_24 = [var_21, var_23]
    var_25 = module_1.Function(var_19, var_24)
    var_26 = 'div'
    var_27 = '@class'
    var_28 = '2'
    var_29 = module_1.parse(var_28)
    var_30 = [var_29]
    var_31 = module_1.Function(var_19, var_30)



# Parsed testcases at query #35
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = '0'
    var_4 = '-1'
    var_5 = 'STRING'
    var_6 = 'test'



# Parsed testcases at query #36
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'IDENT'
    var_3 = 'NUMBER'



# Parsed testcases at query #37
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda : var_14
    var_16 = ()
    var_17 = '1'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda : var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #38
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = [var_5]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = '.baz'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = 'NUMBER'
    var_28 = [var_27]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = '1'
    var_32 = {var_10: var_31}



# Parsed testcases at query #39
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'eq'
    var_2 = 0
    var_3 = '0'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = '//h1'
    var_8 = '5'
    var_9 = parse(var_8)[var_2]
    var_10 = [var_9]
    var_11 = module_1.Function(var_1, var_10)
    var_12 = '//div'
    var_13 = '-1'
    var_14 = parse(var_13)[var_2]
    var_15 = [var_14]
    var_16 = module_1.Function(var_1, var_15)
    var_17 = '//p'
    var_18 = 'eq'
    var_19 = 'invalid'
    var_20 = [var_3]
    var_21 = module_1.Function(var_18, var_20)
    var_22 = '//h1'
    var_23 = var_0.xpath_eq_function(var_7, var_21)



# Parsed testcases at query #40
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = 'Function'
    var_20 = ()
    var_21 = 'argument_types'
    var_22 = 'arguments'
    var_23 = 'STRING'
    var_24 = [var_23]
    var_25 = lambda self: var_24
    var_26 = 'Argument'
    var_27 = ()
    var_28 = 'value'
    var_29 = 'test'
    var_30 = {var_28: var_29}



# Parsed testcases at query #41
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'test'
    var_2 = [var_1]



# Parsed testcases at query #42
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'test text'
    var_3 = 'STRING'
    var_4 = module_0.XPathExpr()
    var_5 = 'title'
    var_6 = 'IDENT'
    var_7 = module_0.XPathExpr()
    var_8 = '1'
    var_9 = 'NUMBER'



# Parsed testcases at query #43
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #44
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'name'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '0'
    var_14 = {var_12: var_13}
    var_15 = 'gt'
    var_16 = ()
    var_17 = [var_7]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = '2'
    var_21 = {var_12: var_20}
    var_22 = '//div'
    var_23 = 'Function'
    var_24 = ()
    var_25 = 'argument_types'
    var_26 = 'arguments'
    var_27 = 'name'
    var_28 = 'STRING'
    var_29 = [var_28]
    var_30 = lambda self: var_29
    var_31 = 'Argument'
    var_32 = ()
    var_33 = 'value'
    var_34 = 'test'
    var_35 = {var_33: var_34}
    var_36 = 'gt'



# Parsed testcases at query #45
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
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
    var_16 = module_0.XPathExpr(var_1, var_2)
    var_17 = ()
    var_18 = 'IDENT'
    var_19 = [var_18]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = {var_13: var_2}
    var_23 = module_0.XPathExpr(var_1, var_2)
    var_24 = ()
    var_25 = [var_8]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '.test'
    var_29 = {var_13: var_28}
    var_30 = module_0.XPathExpr(var_1, var_2)
    var_31 = ()
    var_32 = 'NUMBER'
    var_33 = [var_32]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = '1'
    var_37 = {var_13: var_36}
    var_38 = '//div[@class="foo"]'
    var_39 = module_0.XPathExpr(var_38, var_2)
    var_40 = ()
    var_41 = [var_8]
    var_42 = lambda self: var_41
    var_43 = ()
    var_44 = {var_13: var_14}



# Parsed testcases at query #46
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '.foo:has(".bar")'
    var_2 = '.foo:has(div)'
    var_3 = '.foo:has(".baz")'
    var_4 = 'has'
    var_5 = '1'
    var_6 = [var_5]
    var_7 = module_1.Function(var_4, var_6)
    var_8 = 'NUMBER'
    var_9 = [var_8]



# Parsed testcases at query #47
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '.bar'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = {var_12: var_2}
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Argument'
    var_29 = ()
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}
    var_33 = [var_14]
    var_34 = {var_23: var_27, var_24: var_33}



# Parsed testcases at query #48
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = [var_5, var_5]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = {var_10: var_11}
    var_31 = ()
    var_32 = '2'
    var_33 = {var_10: var_32}
    var_34 = ()
    var_35 = [var_5]
    var_36 = lambda self: var_35
    var_37 = ()
    var_38 = '-1'
    var_39 = {var_10: var_38}
    var_40 = ()
    var_41 = [var_5]
    var_42 = lambda self: var_41
    var_43 = ()
    var_44 = '100'
    var_45 = {var_10: var_44}
    var_46 = ()
    var_47 = [var_5]
    var_48 = lambda self: var_47
    var_49 = ()
    var_50 = '5'
    var_51 = {var_10: var_50}



# Parsed testcases at query #49
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #50
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = ''
    var_3 = module_0.XPathExpr(var_1, var_1, var_2)
    var_4 = 0
    var_5 = module_0.XPathExpr(var_1, var_1, var_2)
    var_6 = 1
    var_7 = module_0.XPathExpr(var_1, var_1, var_2)
    var_8 = 5
    var_9 = 'div'
    var_10 = ''
    var_11 = module_0.XPathExpr(var_9, var_9, var_10)
    var_12 = 'div'
    var_13 = ''
    var_14 = module_0.XPathExpr(var_12, var_12, var_13)



# Parsed testcases at query #51
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'
    var_4 = '-1'



# Parsed testcases at query #52
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = '1'
    var_3 = '0'
    var_4 = '2'
    var_5 = 'div'
    var_6 = '"test"'



# Parsed testcases at query #53
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = 'title'
    var_14 = {var_12: var_13}
    var_15 = '//div'
    var_16 = 'div'
    var_17 = ()
    var_18 = 'IDENT'
    var_19 = [var_18]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = 'content'
    var_23 = {var_12: var_22}
    var_24 = '//p'
    var_25 = 'p'
    var_26 = ()
    var_27 = [var_7]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = ''
    var_31 = {var_12: var_30}
    var_32 = '//span'
    var_33 = 'span'
    var_34 = ()
    var_35 = [var_7]
    var_36 = lambda self: var_35
    var_37 = ()
    var_38 = "it's"
    var_39 = {var_12: var_38}
    var_40 = '//a'
    var_41 = 'a'
    var_42 = ()
    var_43 = 'NUMBER'
    var_44 = [var_43]
    var_45 = lambda self: var_44
    var_46 = ()
    var_47 = '42'
    var_48 = {var_12: var_47}



# Parsed testcases at query #54
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #55
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'title'
    var_5 = [var_4]
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = [var_4]
    var_9 = [var_2]
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = 'h1'
    var_13 = 'NUMBER'
    var_14 = [var_13]
    var_15 = '1'
    var_16 = [var_15]
    var_17 = 'h1'
    var_18 = 'STRING'
    var_19 = [var_18, var_18]
    var_20 = 'a'
    var_21 = 'b'
    var_22 = [var_20, var_21]



# Parsed testcases at query #56
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #57
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = 'class="foo"'
    var_3 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'arguments'
    var_7 = 'argument_types'
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = 'STRING'
    var_14 = [var_13]
    var_15 = lambda self: var_14
    var_16 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_17 = ()
    var_18 = ()
    var_19 = '.baz'
    var_20 = {var_10: var_19}
    var_21 = [var_13]
    var_22 = lambda self: var_21
    var_23 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_24 = ()
    var_25 = ()
    var_26 = {var_10: var_1}
    var_27 = 'IDENT'
    var_28 = [var_27]
    var_29 = lambda self: var_28
    var_30 = 'div'
    var_31 = module_0.XPathExpr(element=var_30)
    var_32 = 'Function'
    var_33 = ()
    var_34 = 'arguments'
    var_35 = 'argument_types'
    var_36 = 'Arg'
    var_37 = ()
    var_38 = 'value'
    var_39 = '123'
    var_40 = {var_38: var_39}
    var_41 = 'NUMBER'
    var_42 = [var_41]
    var_43 = lambda self: var_42



# Parsed testcases at query #58
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = None
    var_2 = lambda : var_1
    var_3 = 'arg'
    var_4 = ()
    var_5 = 'value'
    var_6 = 'type'
    var_7 = '.bar'
    var_8 = 'STRING'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = [var_8]
    var_11 = ()
    var_12 = 'div'
    var_13 = 'IDENT'
    var_14 = {var_5: var_12, var_6: var_13}
    var_15 = [var_13]
    var_16 = ()
    var_17 = '1'
    var_18 = 'NUMBER'
    var_19 = {var_5: var_17, var_6: var_18}
    var_20 = [var_18]
    var_21 = 'arg1'
    var_22 = ()
    var_23 = {var_5: var_7, var_6: var_8}
    var_24 = 'arg2'
    var_25 = ()
    var_26 = '.baz'
    var_27 = {var_5: var_26, var_6: var_8}
    var_28 = [var_8, var_8]
    var_29 = ()
    var_30 = '.bar > .baz'
    var_31 = {var_5: var_30, var_6: var_8}
    var_32 = [var_8]
    var_33 = ()
    var_34 = 'p'
    var_35 = {var_5: var_34, var_6: var_8}
    var_36 = [var_8]
    var_37 = ()
    var_38 = ''
    var_39 = {var_5: var_38, var_6: var_8}
    var_40 = [var_8]



# Parsed testcases at query #59
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"test"'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = 'test'
    var_8 = parse(var_7)[var_2]
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = 'contains'
    var_12 = 0
    var_13 = '123'
    var_14 = parse(var_13)[var_12]
    var_15 = [var_14]
    var_16 = module_1.Function(var_11, var_15)



# Parsed testcases at query #60
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockFunction'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'MockArgument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '3'
    var_12 = {var_10: var_11}



# Parsed testcases at query #61
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '2'
    var_4 = 'div'
    var_5 = '@class'
    var_6 = '1'
    var_7 = 'STRING'
    var_8 = 'test'
    var_9 = 'NUMBER'
    var_10 = [var_9, var_9]
    var_11 = '1'
    var_12 = '2'
    var_13 = [var_11, var_12]



# Parsed testcases at query #62
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = [var_6]
    var_16 = lambda : var_15
    var_17 = ()
    var_18 = {var_11: var_1}
    var_19 = ()
    var_20 = 'IDENT'
    var_21 = [var_20]
    var_22 = lambda : var_21
    var_23 = ()
    var_24 = 'bar'
    var_25 = {var_11: var_24}
    var_26 = 'div'
    var_27 = 'Function'
    var_28 = ()
    var_29 = 'argument_types'
    var_30 = 'arguments'
    var_31 = 'NUMBER'
    var_32 = [var_31]
    var_33 = lambda : var_32
    var_34 = 'Arg'
    var_35 = ()
    var_36 = 'value'
    var_37 = '1'
    var_38 = {var_36: var_37}



# Parsed testcases at query #63
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}



# Parsed testcases at query #64
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '1'
    var_2 = module_0.XPathExpr()
    var_3 = '3'
    var_4 = module_0.XPathExpr()
    var_5 = '0'
    var_6 = module_0.XPathExpr()
    var_7 = 'test'
    var_8 = module_0.XPathExpr()



# Parsed testcases at query #65
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.JQueryTranslator()
    var_2 = 'position() = 1'



# Parsed testcases at query #66
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #67
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = [var_6]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = {var_11: var_1}
    var_19 = ()
    var_20 = 'IDENT'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_11: var_24}
    var_26 = 'div'
    var_27 = 'Function'
    var_28 = ()
    var_29 = 'argument_types'
    var_30 = 'arguments'
    var_31 = 'NUMBER'
    var_32 = [var_31]
    var_33 = lambda self: var_32
    var_34 = 'Argument'
    var_35 = ()
    var_36 = 'value'
    var_37 = '1'
    var_38 = {var_36: var_37}



# Parsed testcases at query #68
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = 'content'
    var_4 = 'IDENT'



# Parsed testcases at query #69
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #70
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = ''
    var_4 = module_0.XPathExpr(var_1, var_2, var_3)
    var_5 = 'obj'
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'value'
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr(var_1, var_2, var_3)
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = 'span'
    var_19 = {var_11: var_18}
    var_20 = '//div'
    var_21 = 'div'
    var_22 = ''
    var_23 = module_0.XPathExpr(var_20, var_21, var_22)
    var_24 = 'obj'
    var_25 = 'argument_types'
    var_26 = 'arguments'
    var_27 = 'NUMBER'
    var_28 = [var_27]
    var_29 = lambda self: var_28
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}



# Parsed testcases at query #71
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = "@class='foo'"
    var_3 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda : var_9
    var_11 = 'Arg'
    var_12 = ()
    var_13 = 'value'
    var_14 = '.bar'
    var_15 = {var_13: var_14}
    var_16 = module_0.XPathExpr(element=var_1)
    var_17 = ()
    var_18 = [var_8]
    var_19 = lambda : var_18
    var_20 = ()
    var_21 = {var_13: var_1}
    var_22 = module_0.XPathExpr(element=var_1)
    var_23 = ()
    var_24 = [var_8]
    var_25 = lambda : var_24
    var_26 = ()
    var_27 = '#myid'
    var_28 = {var_13: var_27}
    var_29 = module_0.XPathExpr(element=var_1)
    var_30 = ()
    var_31 = 'IDENT'
    var_32 = [var_31]
    var_33 = lambda : var_32
    var_34 = ()
    var_35 = {var_13: var_1}
    var_36 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_37 = ()
    var_38 = [var_8]
    var_39 = lambda : var_38
    var_40 = ()
    var_41 = {var_13: var_14}
    var_42 = 'and'
    var_43 = 'descendant::'
    var_44 = 'div'
    var_45 = module_0.XPathExpr(element=var_44)
    var_46 = 'Function'
    var_47 = ()
    var_48 = 'argument_types'
    var_49 = 'arguments'
    var_50 = 'NUMBER'
    var_51 = [var_50]
    var_52 = lambda : var_51
    var_53 = 'Arg'
    var_54 = ()
    var_55 = 'value'
    var_56 = '1'
    var_57 = {var_55: var_56}
    var_58 = module_0.XPathExpr(element=var_44)
    var_59 = ()
    var_60 = [var_51]
    var_61 = lambda : var_60
    var_62 = ()
    var_63 = ''
    var_64 = {var_56: var_63}



# Parsed testcases at query #72
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = '0'
    var_4 = '-1'
    var_5 = '100'
    var_6 = 'STRING'
    var_7 = 'invalid'



# Parsed testcases at query #73
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 'STRING'
    var_3 = 'title'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = 'IDENT'
    var_8 = (var_7, var_3)
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = 'contains'
    var_12 = 'NUMBER'
    var_13 = '1'
    var_14 = (var_12, var_13)
    var_15 = '2'
    var_16 = (var_12, var_15)
    var_17 = [var_14, var_16]
    var_18 = module_1.Function(var_11, var_17)
    var_19 = 'contains'
    var_20 = 'NUMBER'
    var_21 = '42'
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = module_1.Function(var_19, var_23)



# Parsed testcases at query #74
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '2'
    var_4 = 'STRING'
    var_5 = 'test'
    var_6 = 'NUMBER'
    var_7 = '0'
    var_8 = '1'



# Parsed testcases at query #75
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = 0
    var_3 = 2
    var_4 = -1
    var_5 = 'foo'
    var_6 = 1



# Parsed testcases at query #76
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed[var_2]
    var_6 = var_5.arguments[var_2]
    var_7 = [var_6]
    var_8 = module_1.Function(var_1, var_7)
    var_9 = 'title'
    var_10 = parse(var_9)[var_2]
    var_11 = var_10.parsed[var_2]
    var_12 = var_11.arguments[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = '1'
    var_16 = parse(var_15)[var_2]
    var_17 = var_16.parsed[var_2]
    var_18 = var_17.arguments[var_2]
    var_19 = [var_18]
    var_20 = module_1.Function(var_1, var_19)
    var_21 = var_0.xpath_contains_function(var_1, var_20)



# Parsed testcases at query #77
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'Function'
    var_3 = ()
    var_4 = 'arguments'
    var_5 = 'argument_types'
    var_6 = 'Argument'
    var_7 = ()
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}
    var_11 = 'NUMBER'
    var_12 = [var_11]
    var_13 = lambda self: var_12
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = ()
    var_17 = '5'
    var_18 = {var_8: var_17}
    var_19 = [var_11]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = ()
    var_23 = 'text'
    var_24 = {var_8: var_23}
    var_25 = 'STRING'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = module_0.XPathExpr()
    var_29 = ()
    var_30 = ()
    var_31 = '1'
    var_32 = {var_8: var_31}
    var_33 = ()
    var_34 = '2'
    var_35 = {var_8: var_34}
    var_36 = [var_11, var_11]
    var_37 = lambda self: var_36
    var_38 = module_0.XPathExpr()



# Parsed testcases at query #78
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '3'
    var_4 = 'STRING'
    var_5 = 'test'



# Parsed testcases at query #79
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = module_0.JQueryTranslator()
    var_3 = 'IDENT'
    var_4 = module_0.JQueryTranslator()
    var_5 = 'NUMBER'
    var_6 = 'invalid'



# Parsed testcases at query #80
#--------------------------


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
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '3'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = [var_5, var_5]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = '1'
    var_31 = {var_10: var_30}
    var_32 = ()
    var_33 = '2'
    var_34 = {var_10: var_33}



# Parsed testcases at query #81
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = 'Test the xpath_has_function method of JQueryTranslator.'
    var_1 = module_0.JQueryTranslator()
    var_2 = module_0.XPathExpr()
    var_3 = 'has'
    var_4 = 0
    var_5 = '"test"'
    var_6 = parse(var_5)[var_4]
    var_7 = [var_6]
    var_8 = module_1.Function(var_3, var_7)
    var_9 = var_1.xpath_has_function(var_2, var_8)
    var_10 = 'post_condition'
    var_11 = hasattr(var_9, var_10)
    var_12 = module_0.XPathExpr()
    var_13 = 'test'
    var_14 = parse(var_13)[var_4]
    var_15 = [var_14]
    var_16 = module_1.Function(var_3, var_15)
    var_17 = var_1.xpath_has_function(var_12, var_16)
    var_18 = hasattr(var_17, var_10)
    var_19 = '123'
    var_20 = p(var_19)[var_4]
    var_21 = [var_20]
    var_22 = module_1.Function(var_3, var_21)
    var_23 = module_0.XPathExpr()
    var_24 = var_1.xpath_has_function(var_23, var_22)
    var_25 = module_0.XPathExpr()
    var_26 = '"bar"'
    var_27 = p(var_26)[var_4]
    var_28 = [var_27]
    var_29 = module_1.Function(var_24, var_28)
    var_30 = var_1.xpath_has_function(var_25, var_29)
    var_31 = module_0.XPathExpr()
    var_32 = 'div'
    var_33 = p(var_32)[var_4]
    var_34 = [var_33]
    var_35 = module_1.Function(var_24, var_34)
    var_36 = var_1.xpath_has_function(var_31, var_35)



# Parsed testcases at query #82
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'eq'
    var_2 = 0
    var_3 = '0'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = '5'
    var_8 = parse(var_7)[var_2]
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = '-1'
    var_12 = parse(var_11)[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = 'eq'
    var_16 = 0
    var_17 = '"string"'
    var_18 = parse(var_17)[var_16]
    var_19 = [var_18]
    var_20 = module_1.Function(var_15, var_19)
    var_21 = 'eq'
    var_22 = 0
    var_23 = '1'
    var_24 = parse(var_23)[var_22]
    var_25 = '2'
    var_26 = parse(var_25)[var_22]
    var_27 = [var_24, var_26]
    var_28 = module_1.Function(var_21, var_27)
    var_29 = var_0.xpath_eq_function(var_7, var_28)



# Parsed testcases at query #83
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'gt'
    var_3 = '0'
    var_4 = [var_3]
    var_5 = module_1.Function(var_2, var_4)
    var_6 = 0
    var_7 = parse(var_3)[var_6]
    var_8 = var_7.parsed_selectors[var_6]
    var_9 = var_8.pseudo_class.arguments[var_6]
    var_10 = 'NUMBER'
    var_11 = [var_10]
    var_12 = var_0.xpath_gt_function(var_1, var_5)
    var_13 = str(var_12)
    assert var_13 == '*[position() > 1]'
    var_14 = module_0.XPathExpr()
    var_15 = '1'
    var_16 = [var_15]
    var_17 = module_1.Function(var_2, var_16)
    var_18 = parse(var_15)[var_6]
    var_19 = var_18.parsed_selectors[var_6]
    var_20 = var_19.pseudo_class.arguments[var_6]
    var_21 = [var_10]
    var_22 = var_0.xpath_gt_function(var_14, var_17)
    var_23 = str(var_22)
    assert var_23 == '*[position() > 2]'
    var_24 = 'foo'
    var_25 = [var_24]
    var_26 = module_1.Function(var_2, var_25)
    var_27 = parse(var_24)[var_6]
    var_28 = var_27.parsed_selectors[var_6]
    var_29 = var_28.pseudo_class.arguments[var_6]
    var_30 = 'IDENT'
    var_31 = [var_30]
    var_32 = module_0.XPathExpr()
    var_33 = var_0.xpath_gt_function(var_32, var_26)



# Parsed testcases at query #84
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '2'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = [var_5, var_5]
    var_28 = lambda self: var_27
    var_29 = '1'
    var_30 = [var_11, var_29]
    var_31 = {var_3: var_28, var_4: var_30}



# Parsed testcases at query #85
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = 'span'
    var_20 = {var_11: var_19}
    var_21 = 'div'
    var_22 = 'Function'
    var_23 = ()
    var_24 = 'argument_types'
    var_25 = 'arguments'
    var_26 = 'NUMBER'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = 'Arg'
    var_30 = ()
    var_31 = 'value'
    var_32 = '5'
    var_33 = {var_31: var_32}



# Parsed testcases at query #86
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = 1
    var_3 = 'invalid'
    var_4 = [var_3]



# Parsed testcases at query #87
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}



# Parsed testcases at query #88
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = {var_11: var_1}
    var_20 = 'div'
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Arg'
    var_29 = ()
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}



# Parsed testcases at query #89
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = [var_6]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '0'
    var_20 = {var_11: var_19}
    var_21 = module_0.XPathExpr()
    var_22 = ()
    var_23 = [var_6]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = '-1'
    var_27 = {var_11: var_26}



# Parsed testcases at query #90
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()



# Parsed testcases at query #91
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #92
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #93
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
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
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = 'div'
    var_20 = {var_11: var_19}
    var_21 = '//div'
    var_22 = 'Function'
    var_23 = ()
    var_24 = 'argument_types'
    var_25 = 'arguments'
    var_26 = 'NUMBER'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = 'Arg'
    var_30 = ()
    var_31 = 'value'
    var_32 = '1'
    var_33 = {var_31: var_32}



# Parsed testcases at query #94
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = [var_5]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = '.baz'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = 'NUMBER'
    var_28 = [var_27]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = '42'
    var_32 = {var_10: var_31}
    var_33 = ()
    var_34 = [var_5]
    var_35 = lambda self: var_34
    var_36 = ()
    var_37 = 'div.bar'
    var_38 = {var_10: var_37}



# Parsed testcases at query #95
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = lambda : var_3
    var_5 = 'content'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = lambda : var_7
    var_9 = '123'
    var_10 = 'NUMBER'
    var_11 = [var_10]
    var_12 = lambda : var_11
    var_13 = ''
    var_14 = [var_2]
    var_15 = lambda : var_14



# Parsed testcases at query #96
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 3'
    var_3 = 'position() < 1'
    var_4 = 'position() < 0'
    var_5 = 'STRING'



# Parsed testcases at query #97
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = 1
    var_3 = 5



# Parsed testcases at query #98
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '@class="foo"'
    var_3 = module_0.XPathExpr(element=var_1, condition=var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Argument'
    var_12 = ()
    var_13 = 'value'
    var_14 = '.bar'
    var_15 = {var_13: var_14}
    var_16 = module_0.XPathExpr(element=var_1)
    var_17 = ()
    var_18 = 'IDENT'
    var_19 = [var_18]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = {var_13: var_1}
    var_23 = 'div'
    var_24 = module_0.XPathExpr(element=var_23)
    var_25 = 'Function'
    var_26 = ()
    var_27 = 'argument_types'
    var_28 = 'arguments'
    var_29 = 'NUMBER'
    var_30 = [var_29]
    var_31 = lambda self: var_30
    var_32 = 'Argument'
    var_33 = ()
    var_34 = 'value'
    var_35 = '1'
    var_36 = {var_34: var_35}



# Parsed testcases at query #99
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = module_0.XPathExpr()
    var_4 = '2'
    var_5 = module_0.XPathExpr()
    var_6 = '-1'
    var_7 = module_0.XPathExpr()
    var_8 = var_0.xpath_gt_function(var_7, var_4)



# Parsed testcases at query #100
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = 'IDENT'
    var_4 = 'div'
    var_5 = 'NUMBER'
    var_6 = '1'
    var_7 = 'position() = 1'
    var_8 = '.test'



# Parsed testcases at query #101
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = '"test"'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = 'STRING'
    var_6 = module_1.Function(var_1, var_4, var_5)
    var_7 = 'test'
    var_8 = module_1.parse(var_7)
    var_9 = [var_8]
    var_10 = 'IDENT'
    var_11 = module_1.Function(var_1, var_9, var_10)
    var_12 = '123'
    var_13 = module_1.parse(var_12)
    var_14 = [var_13]
    var_15 = 'NUMBER'
    var_16 = module_1.Function(var_1, var_14, var_15)
    var_17 = var_0.xpath_contains_function(var_1, var_16)



# Parsed testcases at query #102
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '3'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #103
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'h1'
    var_2 = 'Argument'
    var_3 = ()
    var_4 = 'value'
    var_5 = 'type'
    var_6 = '0'
    var_7 = 'NUMBER'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = [var_7]
    var_10 = ()
    var_11 = '2'
    var_12 = {var_4: var_11, var_5: var_7}
    var_13 = ()
    var_14 = 'test'
    var_15 = 'STRING'
    var_16 = {var_4: var_14, var_5: var_15}
    var_17 = [var_15]



# Parsed testcases at query #104
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//p'
    var_2 = 'p'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '2'
    var_14 = {var_12: var_13}



# Parsed testcases at query #105
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = '1'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = module_1.Function(var_1, var_4)
    var_6 = module_0.XPathExpr()
    var_7 = var_0.xpath_gt_function(var_6, var_5)
    var_8 = str(var_7)
    var_9 = '0'
    var_10 = module_1.parse(var_9)
    var_11 = [var_10]
    var_12 = module_1.Function(var_1, var_11)
    var_13 = module_0.XPathExpr()
    var_14 = var_0.xpath_gt_function(var_13, var_12)
    var_15 = str(var_14)
    var_16 = '-1'
    var_17 = module_1.parse(var_16)
    var_18 = [var_17]
    var_19 = module_1.Function(var_1, var_18)
    var_20 = module_0.XPathExpr()
    var_21 = var_0.xpath_gt_function(var_20, var_19)
    var_22 = str(var_21)
    var_23 = 'test'
    var_24 = module_1.parse(var_23)
    var_25 = [var_24]
    var_26 = module_1.Function(var_1, var_25)
    var_27 = module_0.XPathExpr()
    var_28 = var_0.xpath_gt_function(var_27, var_26)
    var_29 = module_1.parse(var_28)
    var_30 = '2'
    var_31 = module_1.parse(var_30)
    var_32 = [var_29, var_31]
    var_33 = module_1.Function(var_27, var_32)
    var_34 = module_0.XPathExpr()
    var_35 = var_0.xpath_gt_function(var_34, var_33)



# Parsed testcases at query #106
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()
    var_4 = module_0.XPathExpr()



# Parsed testcases at query #107
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '2'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #108
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = None
    var_3 = lambda : var_2
    var_4 = var_0.xpath_has_function(var_1, var_3)
    var_5 = module_0.XPathExpr()
    var_6 = 'STRING'
    var_7 = '.bar'
    var_8 = module_0.XPathExpr()
    var_9 = 'IDENT'
    var_10 = 'div'
    var_11 = module_0.XPathExpr()
    var_12 = 'NUMBER'
    var_13 = '1'
    var_14 = module_0.XPathExpr()



# Parsed testcases at query #109
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}



# Parsed testcases at query #110
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'NUMBER'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Argument'
    var_12 = ()
    var_13 = 'value'
    var_14 = '1'
    var_15 = {var_13: var_14}
    var_16 = '//p'
    var_17 = 'p'
    var_18 = module_0.XPathExpr(var_16, var_17)
    var_19 = ()
    var_20 = [var_8]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '0'
    var_24 = {var_13: var_23}
    var_25 = '//div'
    var_26 = 'div'
    var_27 = module_0.XPathExpr(var_25, var_26)
    var_28 = ()
    var_29 = [var_8]
    var_30 = lambda self: var_29
    var_31 = ()
    var_32 = '-1'
    var_33 = {var_13: var_32}
    var_34 = '//span'
    var_35 = 'span'
    var_36 = module_0.XPathExpr(var_34, var_35)
    var_37 = ()
    var_38 = 'STRING'
    var_39 = [var_38]
    var_40 = lambda self: var_39
    var_41 = ()
    var_42 = 'test'
    var_43 = {var_13: var_42}



# Parsed testcases at query #111
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'NUMBER'
    var_3 = module_0.XPathExpr()



# Parsed testcases at query #112
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = {}
    var_4 = 'NUMBER'
    var_5 = [var_4]
    var_6 = 'Argument'
    var_7 = ()
    var_8 = 'value'
    var_9 = '0'
    var_10 = {var_8: var_9}
    var_11 = ()
    var_12 = {}
    var_13 = [var_4]
    var_14 = ()
    var_15 = '1'
    var_16 = {var_8: var_15}
    var_17 = ()
    var_18 = {}
    var_19 = [var_4]
    var_20 = ()
    var_21 = '5'
    var_22 = {var_8: var_21}
    var_23 = ()
    var_24 = {}
    var_25 = [var_4]
    var_26 = ()
    var_27 = '-1'
    var_28 = {var_8: var_27}
    var_29 = ()
    var_30 = {}
    var_31 = 'STRING'
    var_32 = [var_31]
    var_33 = ()
    var_34 = 'test'
    var_35 = {var_8: var_34}
    var_36 = ()
    var_37 = {}
    var_38 = [var_4, var_4]
    var_39 = ()
    var_40 = {var_8: var_9}
    var_41 = ()
    var_42 = {var_8: var_15}



# Parsed testcases at query #113
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'STRING'



# Parsed testcases at query #114
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = module_0.XPathExpr()
    var_4 = '1'
    var_5 = module_0.XPathExpr()
    var_6 = '5'
    var_7 = module_0.XPathExpr()
    var_8 = '-1'
    var_9 = []
    var_10 = module_0.XPathExpr()
    var_11 = var_0.xpath_eq_function(var_10, var_4)



# Parsed testcases at query #115
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 'Argument'
    var_3 = ()
    var_4 = 'type'
    var_5 = 'value'
    var_6 = 'STRING'
    var_7 = 'title'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = '//h1'
    var_10 = 'h1'
    var_11 = ()
    var_12 = 'IDENT'
    var_13 = 'text'
    var_14 = {var_4: var_12, var_5: var_13}
    var_15 = '//p'
    var_16 = 'p'
    var_17 = ()
    var_18 = 'NUMBER'
    var_19 = '123'
    var_20 = {var_4: var_18, var_5: var_19}
    var_21 = '//div'
    var_22 = 'div'



# Parsed testcases at query #116
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #117
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed_tree
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = 'title'
    var_9 = parse(var_8)[var_2]
    var_10 = var_9.parsed_tree
    var_11 = [var_10]
    var_12 = module_1.Function(var_1, var_11)
    var_13 = '1'
    var_14 = parse(var_13)[var_2]
    var_15 = var_14.parsed_tree
    var_16 = [var_15]
    var_17 = module_1.Function(var_1, var_16)
    var_18 = var_0.xpath_contains_function(var_1, var_17)



# Parsed testcases at query #118
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = 'eq'
    var_4 = 'NUMBER'
    var_5 = '0'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = module_1.Function(var_3, var_7)
    var_9 = '//h1'
    var_10 = 'h1'
    var_11 = '3'
    var_12 = (var_4, var_11)
    var_13 = [var_12]
    var_14 = module_1.Function(var_3, var_13)
    var_15 = '//p'
    var_16 = 'p'
    var_17 = '-1'
    var_18 = (var_4, var_17)
    var_19 = [var_18]
    var_20 = module_1.Function(var_3, var_19)
    var_21 = '//span'
    var_22 = 'span'
    var_23 = (var_4, var_5)
    var_24 = [var_23]
    var_25 = module_1.Function(var_3, var_24)
    var_26 = 'STRING'
    var_27 = 'invalid'
    var_28 = (var_26, var_27)
    var_29 = [var_28]
    var_30 = module_1.Function(var_3, var_29)
    var_31 = '1'
    var_32 = (var_4, var_31)
    var_33 = '2'
    var_34 = (var_4, var_33)
    var_35 = [var_32, var_34]
    var_36 = module_1.Function(var_3, var_35)
    var_37 = '999'
    var_38 = (var_4, var_37)
    var_39 = [var_38]
    var_40 = module_1.Function(var_3, var_39)



# Parsed testcases at query #119
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '1'
    var_4 = [var_1]
    var_5 = '0'
    var_6 = [var_1]
    var_7 = '-1'
    var_8 = [var_1]
    var_9 = '100'
    var_10 = 'STRING'
    var_11 = [var_10]
    var_12 = 'test'
    var_13 = 'NUMBER'
    var_14 = [var_13, var_13]
    var_15 = '1, 2'



# Parsed testcases at query #120
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '0'
    var_3 = module_0.XPathExpr()
    var_4 = '2'
    var_5 = module_0.XPathExpr()
    var_6 = '-1'
    var_7 = module_0.XPathExpr()



# Parsed testcases at query #121
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = [var_6]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '0'
    var_20 = {var_11: var_19}
    var_21 = module_0.XPathExpr()
    var_22 = ()
    var_23 = 'STRING'
    var_24 = [var_23]
    var_25 = lambda self: var_24
    var_26 = ()
    var_27 = 'test'
    var_28 = {var_11: var_27}



# Parsed testcases at query #122
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '1'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #123
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'content'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Argument'
    var_28 = ()
    var_29 = 'value'
    var_30 = '1'
    var_31 = {var_29: var_30}



# Parsed testcases at query #124
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'
    var_4 = '-1'
    var_5 = 'test'



# Parsed testcases at query #125
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
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
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'content'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '5'
    var_29 = {var_11: var_28}



# Parsed testcases at query #126
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '0'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = [var_7]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '2'
    var_20 = {var_12: var_19}
    var_21 = ()
    var_22 = [var_7]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '-1'
    var_26 = {var_12: var_25}



# Parsed testcases at query #127
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = 'Test xpath_contains_function with various inputs.'
    var_1 = module_0.JQueryTranslator()
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
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = 'XPathExpr'
    var_15 = ()
    var_16 = 'post_condition'
    var_17 = 'add_post_condition'
    var_18 = None
    var_19 = lambda self, cond: setattr(self, var_16, cond)
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = ()
    var_22 = 'IDENT'
    var_23 = [var_22]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = 'content'
    var_27 = {var_11: var_26}
    var_28 = ()
    var_29 = lambda self, cond: setattr(self, var_16, cond)
    var_30 = {var_16: var_18, var_17: var_29}
    var_31 = ()
    var_32 = 'NUMBER'
    var_33 = [var_32]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = '42'
    var_37 = {var_11: var_36}



# Parsed testcases at query #128
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'test'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '42'
    var_26 = {var_10: var_25}
    var_27 = ()
    var_28 = [var_5, var_5]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = 'foo'
    var_32 = {var_10: var_31}
    var_33 = ()
    var_34 = 'bar'
    var_35 = {var_10: var_34}



# Parsed testcases at query #129
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = ':gt(2)'
    var_3 = parse(var_2)[var_1]
    var_4 = ':gt(0)'
    var_5 = parse(var_4)[var_1]
    var_6 = ':gt(-1)'
    var_7 = parse(var_6)[var_1]



# Parsed testcases at query #130
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'text'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Argument'
    var_28 = ()
    var_29 = 'value'
    var_30 = '1'
    var_31 = {var_29: var_30}
    var_32 = 'Function'
    var_33 = ()
    var_34 = 'argument_types'
    var_35 = 'arguments'
    var_36 = 'STRING'
    var_37 = [var_36, var_36]
    var_38 = lambda self: var_37
    var_39 = 'Argument'
    var_40 = ()
    var_41 = 'value'
    var_42 = 'a'
    var_43 = {var_41: var_42}
    var_44 = ()
    var_45 = 'b'
    var_46 = {var_41: var_45}



# Parsed testcases at query #131
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'has'
    var_2 = 'div'
    var_3 = '.bar'
    var_4 = 'descendant::*[contains'
    var_5 = 'descendant::*[@class'
    var_6 = 'div.baz'
    var_7 = 'p'
    var_8 = 'has'
    var_9 = '1'



# Parsed testcases at query #132
#--------------------------


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
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = [var_5]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '100'
    var_30 = {var_10: var_29}
    var_31 = ()
    var_32 = 'STRING'
    var_33 = [var_32]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = 'test'
    var_37 = {var_10: var_36}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "@type = 'text'"



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'test'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Argument'
    var_28 = ()
    var_29 = 'value'
    var_30 = '1'
    var_31 = {var_29: var_30}



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = 'post_condition'
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda : var_16
    var_18 = ()
    var_19 = 'div'
    var_20 = {var_10: var_19}
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda : var_26
    var_28 = 'Arg'
    var_29 = ()
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}
    var_33 = ()
    var_34 = [var_25]
    var_35 = lambda : var_34
    var_36 = ()
    var_37 = 'div.bar'
    var_38 = {var_30: var_37}
    var_39 = '/test'
    var_40 = ()
    var_41 = [var_25]
    var_42 = lambda : var_41
    var_43 = ()
    var_44 = '.foo'
    var_45 = {var_30: var_44}
    var_46 = 'descendant::'



# Parsed testcases at query #4
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()



# Parsed testcases at query #5
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'input'
    var_2 = 'select'
    var_3 = 'textarea'
    var_4 = 'button'



# Parsed testcases at query #6
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '0'
    var_14 = {var_12: var_13}
    var_15 = module_0.XPathExpr(var_1)
    var_16 = ()
    var_17 = [var_7]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = '1'
    var_21 = {var_12: var_20}
    var_22 = module_0.XPathExpr(var_1)
    var_23 = ()
    var_24 = 'STRING'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = 'test'
    var_29 = {var_12: var_28}



# Parsed testcases at query #7
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}
    var_32 = ()
    var_33 = [var_5, var_5]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = {var_10: var_11}
    var_37 = ()
    var_38 = '2'
    var_39 = {var_10: var_38}



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '*'
    var_2 = "(name(.) = 'input' or name(.) = 'select') or (name(.) = 'textarea' or name(.) = 'button')"



# Parsed testcases at query #9
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'input'
    var_2 = 'select'
    var_3 = 'textarea'
    var_4 = 'button'
    var_5 = -1
    var_6 = '['
    var_7 = var_21.split(var_6)[var_5]
    var_8 = ']'



# Parsed testcases at query #10
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #11
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '2'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//p'
    var_2 = 'p'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'NUMBER'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Argument'
    var_12 = ()
    var_13 = 'value'
    var_14 = '0'
    var_15 = {var_13: var_14}
    var_16 = str(var_3)
    assert var_16 == '//p[position() > 1]'
    var_17 = '//div'
    var_18 = 'div'
    var_19 = module_0.XPathExpr(var_17, var_18)
    var_20 = ()
    var_21 = [var_8]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = '2'
    var_25 = {var_13: var_24}
    var_26 = str(var_19)
    assert var_26 == '//div[position() > 3]'
    var_27 = '//span'
    var_28 = 'span'
    var_29 = module_0.XPathExpr(var_27, var_28)
    var_30 = ()
    var_31 = [var_8]
    var_32 = lambda self: var_31
    var_33 = ()
    var_34 = '-1'
    var_35 = {var_13: var_34}
    var_36 = str(var_29)
    assert var_36 == '//span[position() > 0]'
    var_37 = '//li'
    var_38 = 'li'
    var_39 = module_0.XPathExpr(var_37, var_38)
    var_40 = ()
    var_41 = [var_8]
    var_42 = lambda self: var_41
    var_43 = ()
    var_44 = '1'
    var_45 = {var_13: var_44}
    var_46 = str(var_39)
    assert var_46 == '//li[position() < 5][position() > 2]'
    var_47 = '//a'
    var_48 = 'a'
    var_49 = module_0.XPathExpr(var_47, var_48)
    var_50 = 'Function'
    var_51 = ()
    var_52 = 'argument_types'
    var_53 = 'arguments'
    var_54 = 'STRING'
    var_55 = [var_54]
    var_56 = lambda self: var_55
    var_57 = 'Argument'
    var_58 = ()
    var_59 = 'value'
    var_60 = 'test'
    var_61 = {var_59: var_60}



# Parsed testcases at query #13
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '2'
    var_4 = [var_3]
    var_5 = [var_1]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = '-1'
    var_10 = [var_9]
    var_11 = 'STRING'
    var_12 = [var_11]
    var_13 = 'test'
    var_14 = [var_13]
    var_15 = [var_1, var_1]
    var_16 = '1'
    var_17 = [var_16, var_3]
    var_18 = [var_1]
    var_19 = '3.5'
    var_20 = [var_19]



# Parsed testcases at query #14
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'value'
    var_6 = 'STRING'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'Arg'
    var_10 = ()
    var_11 = '.bar'
    var_12 = {var_5: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_5: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '42'
    var_26 = {var_5: var_25}
    var_27 = ()
    var_28 = [var_6, var_6]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = 'foo'
    var_32 = {var_5: var_31}
    var_33 = ()
    var_34 = 'bar'
    var_35 = {var_5: var_34}
    var_36 = 'foo, bar'



# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'title'
    var_2 = 'STRING'
    var_3 = module_0.XPathExpr()
    var_4 = 'IDENT'
    var_5 = module_0.XPathExpr()
    var_6 = '1'
    var_7 = 'NUMBER'
    var_8 = module_0.XPathExpr()



# Parsed testcases at query #16
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '1'
    var_26 = {var_10: var_25}
    var_27 = ()
    var_28 = [var_5, var_5]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = 'foo'
    var_32 = {var_10: var_31}



# Parsed testcases at query #17
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = 'div'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'NUMBER'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = '5'
    var_25 = {var_10: var_24}



# Parsed testcases at query #18
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = ''
    var_4 = False
    var_5 = module_0.XPathExpr(var_1, var_2, var_3, var_4)
    var_6 = 'Function'
    var_7 = ()
    var_8 = 'arguments'
    var_9 = 'argument_types'
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = 'type'
    var_14 = 'title'
    var_15 = 'STRING'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = [var_15]
    var_18 = lambda self: var_17
    var_19 = '//h1'
    var_20 = 'h1'
    var_21 = module_0.XPathExpr(var_19, var_20, var_3, var_4)
    var_22 = ()
    var_23 = ()
    var_24 = 'content'
    var_25 = 'IDENT'
    var_26 = {var_12: var_24, var_13: var_25}
    var_27 = [var_25]
    var_28 = lambda self: var_27
    var_29 = '//*'
    var_30 = '*'
    var_31 = module_0.XPathExpr(var_29, var_30, var_3, var_4)
    var_32 = ()
    var_33 = ()
    var_34 = 123
    var_35 = 'NUMBER'
    var_36 = {var_12: var_34, var_13: var_35}
    var_37 = [var_35]
    var_38 = lambda self: var_37
    var_39 = module_0.XPathExpr(var_29, var_30, var_3, var_4)
    var_40 = ()
    var_41 = ()
    var_42 = 'text'
    var_43 = {var_12: var_42, var_13: var_15}
    var_44 = ()
    var_45 = 'extra'
    var_46 = {var_12: var_45, var_13: var_15}
    var_47 = [var_15, var_15]
    var_48 = lambda self: var_47



# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = ''
    var_3 = False
    var_4 = module_0.XPathExpr(var_1, var_1, var_2, var_3)
    var_5 = 'Function'
    var_6 = ()
    var_7 = 'argument_types'
    var_8 = 'arguments'
    var_9 = 'NUMBER'
    var_10 = [var_9]
    var_11 = lambda self: var_10
    var_12 = 'Argument'
    var_13 = ()
    var_14 = 'value'
    var_15 = '0'
    var_16 = {var_14: var_15}
    var_17 = module_0.XPathExpr(var_1, var_1, var_2, var_3)
    var_18 = ()
    var_19 = [var_9]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = '3'
    var_23 = {var_14: var_22}
    var_24 = module_0.XPathExpr(var_1, var_1, var_2, var_3)
    var_25 = ()
    var_26 = [var_9]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '-1'
    var_30 = {var_14: var_29}



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockFunction'
    var_2 = ()
    var_3 = {}
    var_4 = 'MockArgument'
    var_5 = ()
    var_6 = 'value'
    var_7 = 'type'
    var_8 = 'title'
    var_9 = 'STRING'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = [var_9]
    var_12 = module_0.XPathExpr()
    var_13 = ()
    var_14 = {}
    var_15 = ()
    var_16 = 'text'
    var_17 = 'IDENT'
    var_18 = {var_6: var_16, var_7: var_17}
    var_19 = [var_17]
    var_20 = module_0.XPathExpr()
    var_21 = ()
    var_22 = {}
    var_23 = ()
    var_24 = '42'
    var_25 = 'NUMBER'
    var_26 = {var_6: var_24, var_7: var_25}
    var_27 = [var_25]
    var_28 = module_0.XPathExpr()
    var_29 = ()
    var_30 = {}
    var_31 = ()
    var_32 = 'hello'
    var_33 = {var_6: var_32, var_7: var_9}
    var_34 = ()
    var_35 = 'world'
    var_36 = {var_6: var_35, var_7: var_9}
    var_37 = [var_9, var_9]
    var_38 = module_0.XPathExpr()
    var_39 = ()
    var_40 = {}
    var_41 = ()
    var_42 = ''
    var_43 = {var_6: var_42, var_7: var_9}
    var_44 = [var_9]
    var_45 = module_0.XPathExpr()



# Parsed testcases at query #21
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'div'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = 0
    var_29 = {var_11: var_28}
    var_30 = module_0.XPathExpr()
    var_31 = ()
    var_32 = [var_6]
    var_33 = lambda self: var_32
    var_34 = ()
    var_35 = '.test'
    var_36 = {var_11: var_35}



# Parsed testcases at query #22
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []



# Parsed testcases at query #23
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
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
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'test'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '42'
    var_29 = {var_11: var_28}
    var_30 = module_0.XPathExpr()
    var_31 = ()
    var_32 = [var_6, var_24]
    var_33 = lambda self: var_32
    var_34 = ()
    var_35 = {var_11: var_20}



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []



# Parsed testcases at query #25
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'MockXPath'
    var_2 = ()
    var_3 = 'post_condition'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = ()
    var_7 = {var_3: var_4}



# Parsed testcases at query #26
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'invalid'
    var_2 = [var_1]



# Parsed testcases at query #27
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '2'
    var_4 = [var_3]
    var_5 = [var_1]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = '-1'
    var_10 = [var_9]
    var_11 = 'STRING'
    var_12 = [var_11]
    var_13 = 'test'
    var_14 = [var_13]
    var_15 = [var_1, var_1]
    var_16 = '1'
    var_17 = [var_16, var_3]



# Parsed testcases at query #28
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '5'
    var_26 = {var_10: var_25}



# Parsed testcases at query #29
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 0
    var_2 = 1
    var_3 = 5



# Parsed testcases at query #30
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'title'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '123'
    var_26 = {var_10: var_25}
    var_27 = ()
    var_28 = [var_5, var_5]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = {var_10: var_11}



# Parsed testcases at query #31
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '1'
    var_4 = '5'
    var_5 = 'STRING'
    var_6 = 'test'



# Parsed testcases at query #32
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = 'title'
    var_9 = parse(var_8)[var_2]
    var_10 = var_9.parsed
    var_11 = [var_10]
    var_12 = module_1.Function(var_1, var_11)
    var_13 = 'contains'
    var_14 = 0
    var_15 = '1'
    var_16 = parse(var_15)[var_14]
    var_17 = var_16.parsed
    var_18 = [var_17]
    var_19 = module_1.Function(var_13, var_18)
    var_20 = '"test"'
    var_21 = parse(var_20)[var_14]
    var_22 = var_21.parsed
    var_23 = [var_22]
    var_24 = module_1.Function(var_13, var_23)
    var_25 = '"first"'
    var_26 = parse(var_25)[var_14]
    var_27 = var_26.parsed
    var_28 = [var_27]
    var_29 = module_1.Function(var_13, var_28)



# Parsed testcases at query #33
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'eq'
    var_2 = '0'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = module_1.Function(var_1, var_4)
    var_6 = '5'
    var_7 = module_1.parse(var_6)
    var_8 = [var_7]
    var_9 = module_1.Function(var_1, var_8)
    var_10 = '-1'
    var_11 = module_1.parse(var_10)
    var_12 = [var_11]
    var_13 = module_1.Function(var_1, var_12)
    var_14 = '"string"'
    var_15 = module_1.parse(var_14)
    var_16 = [var_15]
    var_17 = module_1.Function(var_1, var_16)
    var_18 = var_0.xpath_eq_function(var_1, var_17)
    var_19 = 'ident'
    var_20 = module_1.parse(var_19)
    var_21 = [var_20]
    var_22 = module_1.Function(var_1, var_21)
    var_23 = var_0.xpath_eq_function(var_1, var_22)
    var_24 = '1'
    var_25 = module_1.parse(var_24)
    var_26 = '2'
    var_27 = module_1.parse(var_26)
    var_28 = [var_25, var_27]
    var_29 = module_1.Function(var_1, var_28)
    var_30 = var_0.xpath_eq_function(var_1, var_29)



# Parsed testcases at query #34
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'contains'
    var_3 = 0
    var_4 = '"title"'
    var_5 = parse(var_4)[var_3]
    var_6 = var_5.parsed_selectors[var_3]
    var_7 = var_6.pseudo_class.arguments[var_3]
    var_8 = [var_7]
    var_9 = module_1.Function(var_2, var_8)
    var_10 = module_0.XPathExpr()
    var_11 = 'title'
    var_12 = 'STRING'
    var_13 = module_0.XPathExpr()
    var_14 = 'text'
    var_15 = 'IDENT'
    var_16 = module_0.XPathExpr()
    var_17 = '42'
    var_18 = 'NUMBER'
    var_19 = module_0.XPathExpr()
    var_20 = 'hello'
    var_21 = 'world'



# Parsed testcases at query #35
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'contains'
    var_5 = 0
    var_6 = '"title"'
    var_7 = parse(var_6)[var_5]
    var_8 = [var_7]
    var_9 = module_1.Function(var_4, var_8)
    var_10 = var_0.xpath_contains_function(var_3, var_9)
    var_11 = str(var_10)
    assert var_11 == "//h1[contains(., 'title')]"
    var_12 = module_0.XPathExpr(var_1, var_2)
    var_13 = 'title'
    var_14 = parse(var_13)[var_5]
    var_15 = [var_14]
    var_16 = module_1.Function(var_4, var_15)
    var_17 = var_0.xpath_contains_function(var_12, var_16)
    var_18 = str(var_17)
    assert var_18 == "//h1[contains(., 'title')]"
    var_19 = module_0.XPathExpr(var_1, var_2)
    var_20 = '1'
    var_21 = parse(var_20)[var_5]
    var_22 = [var_21]
    var_23 = module_1.Function(var_4, var_22)
    var_24 = var_0.xpath_contains_function(var_19, var_23)
    var_25 = module_0.XPathExpr(var_24, var_2)
    var_26 = '"a"'
    var_27 = parse(var_26)[var_5]
    var_28 = '"b"'
    var_29 = parse(var_28)[var_5]
    var_30 = [var_27, var_29]
    var_31 = module_1.Function(var_4, var_30)
    var_32 = var_0.xpath_contains_function(var_25, var_31)



# Parsed testcases at query #36
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = None
    var_2 = lambda : var_1
    var_3 = 'arg'
    var_4 = ()
    var_5 = 'value'
    var_6 = '.bar'
    var_7 = {var_5: var_6}
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = ()
    var_11 = 'div'
    var_12 = {var_5: var_11}
    var_13 = 'IDENT'
    var_14 = [var_13]
    var_15 = ()
    var_16 = 'test'
    var_17 = {var_5: var_16}
    var_18 = 'NUMBER'
    var_19 = [var_18]
    var_20 = ()
    var_21 = '.test'
    var_22 = {var_5: var_21}
    var_23 = [var_8]
    var_24 = ()
    var_25 = {var_5: var_21}
    var_26 = [var_8]



# Parsed testcases at query #37
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'NUMBER'
    var_5 = '2'
    var_6 = '//p'
    var_7 = 'p'
    var_8 = module_0.XPathExpr(var_6, var_7)
    var_9 = '0'
    var_10 = 'STRING'
    var_11 = 'invalid'



# Parsed testcases at query #38
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 'eq'
    var_3 = '0'
    var_4 = module_1.parse(var_3)
    var_5 = [var_4]
    var_6 = module_1.Function(var_2, var_5)
    var_7 = var_0.xpath_eq_function(var_1, var_6)
    var_8 = str(var_7)
    assert var_8 == 'descendant-or-self::*[position() = 1]'
    var_9 = module_0.XPathExpr()
    var_10 = '5'
    var_11 = module_1.parse(var_10)
    var_12 = [var_11]
    var_13 = module_1.Function(var_2, var_12)
    var_14 = var_0.xpath_eq_function(var_9, var_13)
    var_15 = str(var_14)
    assert var_15 == 'descendant-or-self::*[position() = 6]'
    var_16 = module_0.XPathExpr()
    var_17 = '"string"'
    var_18 = module_1.parse(var_17)
    var_19 = [var_18]
    var_20 = module_1.Function(var_2, var_19)
    var_21 = var_0.xpath_eq_function(var_16, var_20)
    var_22 = module_0.XPathExpr()
    var_23 = 'identifier'
    var_24 = module_1.parse(var_23)
    var_25 = [var_24]
    var_26 = module_1.Function(var_21, var_25)
    var_27 = var_0.xpath_eq_function(var_22, var_26)
    var_28 = module_0.XPathExpr()
    var_29 = '1'
    var_30 = module_1.parse(var_29)
    var_31 = '2'
    var_32 = module_1.parse(var_31)
    var_33 = [var_30, var_32]
    var_34 = module_1.Function(var_27, var_33)
    var_35 = var_0.xpath_eq_function(var_28, var_34)



# Parsed testcases at query #39
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #40
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = 0
    var_3 = '1'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = '0'
    var_8 = parse(var_7)[var_2]
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = '-1'
    var_12 = parse(var_11)[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = 'gt'
    var_16 = 0
    var_17 = '"string"'
    var_18 = parse(var_17)[var_16]
    var_19 = [var_18]
    var_20 = module_1.Function(var_15, var_19)
    var_21 = var_0.xpath_gt_function(var_7, var_20)



# Parsed testcases at query #41
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'
    var_4 = '-1'
    var_5 = var_0.xpath_eq_function(var_1, var_2)



# Parsed testcases at query #42
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div/p'
    var_2 = 'p'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'NUMBER'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Arg'
    var_12 = ()
    var_13 = 'value'
    var_14 = '0'
    var_15 = {var_13: var_14}
    var_16 = module_0.XPathExpr(var_1, var_2)
    var_17 = ()
    var_18 = [var_8]
    var_19 = lambda self: var_18
    var_20 = ()
    var_21 = '2'
    var_22 = {var_13: var_21}



# Parsed testcases at query #43
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'text'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Arg'
    var_28 = ()
    var_29 = 'value'
    var_30 = '42'
    var_31 = {var_29: var_30}



# Parsed testcases at query #44
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'FakeFunction'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'NUMBER'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'FakeArgument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '100'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = [var_5]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '-1'
    var_30 = {var_10: var_29}
    var_31 = 'div'
    var_32 = '@class'
    var_33 = ()
    var_34 = [var_5]
    var_35 = lambda self: var_34
    var_36 = ()
    var_37 = '5'
    var_38 = {var_10: var_37}



# Parsed testcases at query #45
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '0'
    var_2 = '1'
    var_3 = '5'



# Parsed testcases at query #46
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '2'
    var_4 = [var_3]
    var_5 = [var_1]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = 'NUMBER'
    var_13 = [var_12, var_12]
    var_14 = '1'
    var_15 = '2'
    var_16 = [var_14, var_15]



# Parsed testcases at query #47
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #48
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = 'hello'
    var_9 = parse(var_8)[var_2]
    var_10 = var_9.parsed
    var_11 = [var_10]
    var_12 = module_1.Function(var_1, var_11)
    var_13 = '123'
    var_14 = parse(var_13)[var_2]
    var_15 = var_14.parsed
    var_16 = [var_15]
    var_17 = module_1.Function(var_1, var_16)



# Parsed testcases at query #49
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = 0
    var_3 = '2'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = '0'
    var_8 = parse(var_7)[var_2]
    var_9 = [var_8]
    var_10 = module_1.Function(var_1, var_9)
    var_11 = '-1'
    var_12 = parse(var_11)[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = '"abc"'
    var_16 = parse(var_15)[var_2]
    var_17 = [var_16]
    var_18 = module_1.Function(var_1, var_17)



# Parsed testcases at query #50
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = module_0.XPathExpr()
    var_9 = var_0.xpath_contains_function(var_8, var_7)
    var_10 = str(var_9)
    var_11 = 'title'
    var_12 = parse(var_11)[var_2]
    var_13 = var_12.parsed
    var_14 = [var_13]
    var_15 = module_1.Function(var_1, var_14)
    var_16 = module_0.XPathExpr()
    var_17 = var_0.xpath_contains_function(var_16, var_15)
    var_18 = str(var_17)
    var_19 = []
    var_20 = module_1.Function(var_1, var_19)
    var_21 = module_0.XPathExpr()
    var_22 = var_0.xpath_contains_function(var_21, var_20)
    var_23 = '<div><h1/><h1 class="title">title</h1></div>'
    var_24 = 'h1:contains("title")'
    var_25 = 'class'



# Parsed testcases at query #51
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = '[position() > 1]'
    var_14 = ()
    var_15 = [var_5]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = '2'
    var_19 = {var_10: var_18}
    var_20 = '[position() > 3]'
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'STRING'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Argument'
    var_29 = ()
    var_30 = 'value'
    var_31 = 'text'
    var_32 = {var_30: var_31}
    var_33 = 'position() = 1'
    var_34 = ()
    var_35 = [var_24]
    var_36 = lambda self: var_35
    var_37 = ()
    var_38 = {var_29: var_30}



# Parsed testcases at query #52
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 0
    var_3 = module_0.XPathExpr()
    var_4 = 1
    var_5 = module_0.XPathExpr()
    var_6 = 5
    var_7 = module_0.XPathExpr()



# Parsed testcases at query #53
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #54
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'title'
    var_3 = 'IDENT'
    var_4 = 'content'
    var_5 = 'NUMBER'
    var_6 = ''



# Parsed testcases at query #55
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = 'MockFunction'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'MockArgument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '0'
    var_14 = {var_12: var_13}
    var_15 = module_0.XPathExpr()
    var_16 = ()
    var_17 = 'STRING'
    var_18 = [var_17]
    var_19 = lambda self: var_18
    var_20 = ()
    var_21 = 'test'
    var_22 = {var_12: var_21}
    var_23 = module_0.XPathExpr()
    var_24 = ()
    var_25 = [var_7]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '1.5'
    var_29 = {var_12: var_28}



# Parsed testcases at query #56
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'content'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Argument'
    var_28 = ()
    var_29 = 'value'
    var_30 = '42'
    var_31 = {var_29: var_30}



# Parsed testcases at query #57
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #58
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '0'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = [var_6]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '2'
    var_20 = {var_11: var_19}
    var_21 = module_0.XPathExpr()
    var_22 = ()
    var_23 = [var_6]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = '-1'
    var_27 = {var_11: var_26}
    var_28 = 'div'
    var_29 = module_0.XPathExpr(element=var_28)
    var_30 = ()
    var_31 = [var_6]
    var_32 = lambda self: var_31
    var_33 = ()
    var_34 = '5'
    var_35 = {var_11: var_34}



# Parsed testcases at query #59
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '1'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #60
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'invalid'
    var_2 = [var_1]



# Parsed testcases at query #61
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'content'
    var_19 = {var_10: var_18}



# Parsed testcases at query #62
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'has'
    var_2 = 'STRING'
    var_3 = '"test"'
    var_4 = 0
    var_5 = 'IDENT'
    var_6 = 'div'
    var_7 = '"div.foo"'
    var_8 = 'NUMBER'
    var_9 = '123'



# Parsed testcases at query #63
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = '-1'
    var_4 = '5'
    var_5 = 'STRING'
    var_6 = 'test'



# Parsed testcases at query #64
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_12 = 'bar'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = 'test'
    var_20 = {var_11: var_19}
    var_21 = ()
    var_22 = [var_6]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = 'foo'
    var_26 = {var_11: var_25}
    var_27 = '[descendant::foo]'
    var_28 = 'div'
    var_29 = 'Function'
    var_30 = ()
    var_31 = 'argument_types'
    var_32 = 'arguments'
    var_33 = 'NUMBER'
    var_34 = [var_33]
    var_35 = lambda self: var_34
    var_36 = 'Argument'
    var_37 = ()
    var_38 = 'value'
    var_39 = '1'
    var_40 = {var_38: var_39}



# Parsed testcases at query #65
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = '1'
    var_3 = module_0.XPathExpr()
    var_4 = '0'
    var_5 = module_0.XPathExpr()
    var_6 = '5'



# Parsed testcases at query #66
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 'STRING'
    var_3 = '"title"'
    var_4 = 'IDENT'
    var_5 = 'title'
    var_6 = 'NUMBER'
    var_7 = '123'



# Parsed testcases at query #67
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #68
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()



# Parsed testcases at query #69
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #70
#--------------------------


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
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '-1'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '0'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #71
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = 'class'
    var_14 = 'bar'
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'div'
    var_21 = {var_10: var_20}
    var_22 = ()
    var_23 = [var_5]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = ''
    var_27 = {var_10: var_26}
    var_28 = 'descendant::*'
    var_29 = 'descendant::'
    var_30 = 'Function'
    var_31 = ()
    var_32 = 'argument_types'
    var_33 = 'arguments'
    var_34 = 'NUMBER'
    var_35 = [var_34]
    var_36 = lambda self: var_35
    var_37 = 'Arg'
    var_38 = ()
    var_39 = 'value'
    var_40 = '42'
    var_41 = {var_39: var_40}



# Parsed testcases at query #72
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #73
#--------------------------


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
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}
    var_32 = ()
    var_33 = [var_5, var_5]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = '1'
    var_37 = {var_10: var_36}
    var_38 = ()
    var_39 = {var_10: var_11}



# Parsed testcases at query #74
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(element=var_1)
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '.bar'
    var_14 = {var_12: var_13}
    var_15 = module_0.XPathExpr(element=var_1)
    var_16 = ()
    var_17 = [var_7]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = {var_12: var_1}
    var_21 = module_0.XPathExpr(element=var_1)
    var_22 = ()
    var_23 = [var_7]
    var_24 = lambda self: var_23
    var_25 = ()
    var_26 = '.foo'
    var_27 = {var_12: var_26}
    var_28 = module_0.XPathExpr(element=var_1)
    var_29 = ()
    var_30 = [var_7]
    var_31 = lambda self: var_30
    var_32 = ()
    var_33 = '#myid'
    var_34 = {var_12: var_33}



# Parsed testcases at query #75
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '5'
    var_24 = {var_10: var_23}
    var_25 = 'function'
    var_26 = ()
    var_27 = 'argument_types'
    var_28 = 'arguments'
    var_29 = 'NUMBER'
    var_30 = [var_29]
    var_31 = lambda self: var_30
    var_32 = 'arg'
    var_33 = ()
    var_34 = 'value'
    var_35 = '-1'
    var_36 = {var_34: var_35}
    var_37 = 'function'
    var_38 = ()
    var_39 = 'argument_types'
    var_40 = 'arguments'
    var_41 = 'STRING'
    var_42 = [var_41]
    var_43 = lambda self: var_42
    var_44 = 'arg'
    var_45 = ()
    var_46 = 'value'
    var_47 = 'test'
    var_48 = {var_46: var_47}



# Parsed testcases at query #76
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'NUMBER'
    var_5 = '0'
    var_6 = module_0.XPathExpr(var_1, var_2)
    var_7 = '2'
    var_8 = module_0.XPathExpr(var_1, var_2)
    var_9 = 'STRING'
    var_10 = 'invalid'



# Parsed testcases at query #77
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #78
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ':contains("test text")'
    var_2 = 'contains'
    var_3 = 0
    var_4 = '"test text"'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.parsed_selectors[var_3]
    var_7 = var_6.pseudo_class.arguments[var_3]
    var_8 = [var_7]
    var_9 = module_1.Function(var_2, var_8)
    var_10 = 'test'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.parsed_selectors[var_3]
    var_13 = var_12.pseudo_class.arguments[var_3]
    var_14 = [var_13]
    var_15 = module_1.Function(var_2, var_14)
    var_16 = 'contains'
    var_17 = 0
    var_18 = ':first'
    var_19 = module_1.parse(var_18)
    var_20 = var_19.parsed_selectors[var_17]
    var_21 = var_20.pseudo_class
    var_22 = [var_21]
    var_23 = module_1.Function(var_16, var_22)
    var_24 = 'contains'
    var_25 = 0
    var_26 = '123'
    var_27 = module_1.parse(var_26)
    var_28 = var_27.parsed_selectors[var_25]
    var_29 = var_28.pseudo_class.arguments[var_25]
    var_30 = [var_29]
    var_31 = module_1.Function(var_24, var_30)



# Parsed testcases at query #79
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = 0
    var_3 = '"title"'
    var_4 = parse(var_3)[var_2]
    var_5 = [var_4]
    var_6 = module_1.Function(var_1, var_5)
    var_7 = ':contains(foo)'
    var_8 = parse(var_7)[var_2]
    var_9 = '"a"'
    var_10 = parse(var_9)[var_2]
    var_11 = '"b"'
    var_12 = parse(var_11)[var_2]
    var_13 = [var_10, var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = var_0.xpath_contains_function(var_1, var_14)
    var_16 = '123'
    var_17 = parse(var_16)[var_15]
    var_18 = [var_17]
    var_19 = module_1.Function(var_1, var_18)
    var_20 = var_0.xpath_contains_function(var_1, var_19)



# Parsed testcases at query #80
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = 0
    var_3 = '3'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed_tree
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = '-1'
    var_9 = parse(var_8)[var_2]
    var_10 = var_9.parsed_tree
    var_11 = [var_10]
    var_12 = module_1.Function(var_1, var_11)
    var_13 = '0'
    var_14 = parse(var_13)[var_2]
    var_15 = var_14.parsed_tree
    var_16 = [var_15]
    var_17 = module_1.Function(var_1, var_16)
    var_18 = 'lt'
    var_19 = 0
    var_20 = '"string"'
    var_21 = parse(var_20)[var_19]
    var_22 = var_21.parsed_tree
    var_23 = [var_22]
    var_24 = module_1.Function(var_18, var_23)
    var_25 = var_0.xpath_lt_function(var_8, var_24)
    var_26 = '5'
    var_27 = parse(var_26)[var_19]
    var_28 = var_27.parsed_tree
    var_29 = [var_28]
    var_30 = module_1.Function(var_18, var_29)



# Parsed testcases at query #81
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = '1'
    var_5 = 'NUMBER'
    var_6 = module_0.XPathExpr(var_1, var_2)
    var_7 = '0'
    var_8 = '//div'
    var_9 = 'div'
    var_10 = module_0.XPathExpr(var_8, var_9)
    var_11 = '5'
    var_12 = '//p'
    var_13 = 'p'
    var_14 = module_0.XPathExpr(var_12, var_13)
    var_15 = 'test'
    var_16 = 'STRING'
    var_17 = '//p'
    var_18 = 'p'
    var_19 = module_0.XPathExpr(var_17, var_18)
    var_20 = '1'
    var_21 = 'NUMBER'
    var_22 = '2'
    var_23 = var_0.xpath_lt_function(var_19, var_6)



# Parsed testcases at query #82
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = {var_11: var_1}
    var_20 = 'div'
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Arg'
    var_29 = ()
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}



# Parsed testcases at query #83
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []



# Parsed testcases at query #84
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/div/h1'
    var_2 = 'MockFunction'
    var_3 = ()
    var_4 = 'argument_types'
    var_5 = 'arguments'
    var_6 = 'NUMBER'
    var_7 = [var_6]
    var_8 = lambda self: var_7
    var_9 = 'MockArgument'
    var_10 = ()
    var_11 = 'value'
    var_12 = '2'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = 'STRING'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = '0'
    var_20 = {var_11: var_19}



# Parsed testcases at query #85
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = '2'
    var_3 = '0'
    var_4 = 'abc'



# Parsed testcases at query #86
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()



# Parsed testcases at query #87
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = 'Test the xpath_has_function method of JQueryTranslator.'
    var_1 = module_0.JQueryTranslator()
    var_2 = module_0.XPathExpr()
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '.bar'
    var_14 = {var_12: var_13}
    var_15 = module_0.XPathExpr()
    var_16 = ()
    var_17 = 'IDENT'
    var_18 = [var_17]
    var_19 = lambda self: var_18
    var_20 = ()
    var_21 = 'div'
    var_22 = {var_12: var_21}
    var_23 = module_0.XPathExpr()
    var_24 = ()
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '1'
    var_30 = {var_12: var_29}
    var_31 = module_0.XPathExpr()
    var_32 = ()
    var_33 = [var_7, var_7]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = 'test'
    var_37 = {var_12: var_36}
    var_38 = ()
    var_39 = 'test2'
    var_40 = {var_12: var_39}



# Parsed testcases at query #88
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ':has(".bar")'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2[var_3]
    var_5 = var_4.pseudo_element
    var_6 = module_0.XPathExpr()
    var_7 = var_0.xpath_has_function(var_6, var_5)
    var_8 = module_0.XPathExpr()
    var_9 = var_0.xpath_has_function(var_8, var_5)
    var_10 = 'has'
    var_11 = []
    var_12 = module_0.XPathExpr()
    var_13 = 'Arg'
    var_14 = ()
    var_15 = 'value'
    var_16 = 'type'
    var_17 = 'div'
    var_18 = 'IDENT'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.XPathExpr()
    var_21 = var_0.xpath_has_function(var_20, var_5)
    var_22 = 'descendant::'



# Parsed testcases at query #89
#--------------------------


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
    var_11 = '0'
    var_12 = {var_10: var_11}

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
    var_12 = {var_10: var_11}



# Parsed testcases at query #90
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'has'
    var_2 = 0
    var_3 = '".bar"'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed_selectors[var_2]
    var_6 = [var_5]
    var_7 = module_1.Function(var_1, var_6)
    var_8 = '.bar'
    var_9 = 'bar'
    var_10 = 'div'
    var_11 = parse(var_10)[var_2]
    var_12 = var_11.parsed_selectors[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = ':first'
    var_16 = parse(var_15)[var_2]
    var_17 = var_16.parsed_selectors[var_2]
    var_18 = [var_17]
    var_19 = module_1.Function(var_1, var_18)
    var_20 = var_0.xpath_has_function(var_1, var_19)



# Parsed testcases at query #91
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = module_0.XPathExpr()
    var_3 = module_0.XPathExpr()
    var_4 = module_0.XPathExpr()
    var_5 = module_0.XPathExpr()



# Parsed testcases at query #92
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = '2'
    var_3 = '0'
    var_4 = 'string_arg'
    var_5 = '1'



# Parsed testcases at query #93
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 2
    var_2 = 'h1'
    var_3 = 1
    var_4 = 'div'
    var_5 = '@class'
    var_6 = 0
    var_7 = -1



# Parsed testcases at query #94
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'STRING'



# Parsed testcases at query #95
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = ':contains("title")'
    var_2 = 0
    var_3 = ':contains(title)'
    var_4 = ':contains("a")'
    var_5 = ':contains("123")'
    var_6 = ':contains("")'
    var_7 = ':contains("hello world")'
    var_8 = []



# Parsed testcases at query #96
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = []



# Parsed testcases at query #97
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'title'
    var_3 = 'IDENT'
    var_4 = 'test'
    var_5 = 'NUMBER'
    var_6 = 5
    var_7 = ''



# Parsed testcases at query #98
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #99
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.bar'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = 'Function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'Arg'
    var_28 = ()
    var_29 = 'value'
    var_30 = '42'
    var_31 = {var_29: var_30}



# Parsed testcases at query #100
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '3'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #101
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() < 2'
    var_5 = [var_1]
    var_6 = '2'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = 'foo'



# Parsed testcases at query #102
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = 0
    var_3 = '1'
    var_4 = parse(var_3)[var_2]
    var_5 = var_4.parsed_selectors[var_2]
    var_6 = var_5.pseudo_class.arguments[var_2]
    var_7 = [var_6]
    var_8 = module_1.Function(var_1, var_7)
    var_9 = '5'
    var_10 = parse(var_9)[var_2]
    var_11 = var_10.parsed_selectors[var_2]
    var_12 = var_11.pseudo_class.arguments[var_2]
    var_13 = [var_12]
    var_14 = module_1.Function(var_1, var_13)
    var_15 = 'lt'
    var_16 = 0
    var_17 = ':contains("test")'
    var_18 = parse(var_17)[var_16]
    var_19 = var_18.parsed_selectors[var_16]
    var_20 = var_19.pseudo_class.arguments[var_16]
    var_21 = [var_20]
    var_22 = module_1.Function(var_15, var_21)
    var_23 = var_0.xpath_lt_function(var_1, var_22)



# Parsed testcases at query #103
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = 'Function'
    var_26 = ()
    var_27 = 'argument_types'
    var_28 = 'arguments'
    var_29 = 'STRING'
    var_30 = [var_29]
    var_31 = lambda self: var_30
    var_32 = 'Argument'
    var_33 = ()
    var_34 = 'value'
    var_35 = 'test'
    var_36 = {var_34: var_35}



# Parsed testcases at query #104
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda : var_8
    var_10 = 'Argument'
    var_11 = ()
    var_12 = 'value'
    var_13 = '1'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = [var_7]
    var_17 = lambda : var_16
    var_18 = ()
    var_19 = '0'
    var_20 = {var_12: var_19}
    var_21 = ()
    var_22 = [var_7]
    var_23 = lambda : var_22
    var_24 = ()
    var_25 = '5'
    var_26 = {var_12: var_25}
    var_27 = '//h1'
    var_28 = 'h1'
    var_29 = 'Function'
    var_30 = ()
    var_31 = 'argument_types'
    var_32 = 'arguments'
    var_33 = 'STRING'
    var_34 = [var_33]
    var_35 = lambda : var_34
    var_36 = 'invalid'
    var_37 = [var_36]
    var_38 = {var_31: var_35, var_32: var_37}



# Parsed testcases at query #105
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'NUMBER'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = '0'
    var_14 = {var_12: var_13}
    var_15 = '//p'
    var_16 = 'p'
    var_17 = ()
    var_18 = [var_7]
    var_19 = lambda self: var_18
    var_20 = ()
    var_21 = '2'
    var_22 = {var_12: var_21}
    var_23 = '//span'
    var_24 = 'span'
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_12: var_30}



# Parsed testcases at query #106
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = '2'
    var_3 = '0'
    var_4 = 'invalid'



# Parsed testcases at query #107
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = '-1'
    var_4 = '0'
    var_5 = 'STRING'
    var_6 = 'test'
    var_7 = 'NUMBER'
    var_8 = '1'
    var_9 = '2'



# Parsed testcases at query #108
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(element=var_1)
    var_3 = '.bar'
    var_4 = [var_3]
    var_5 = 'STRING'
    var_6 = module_0.XPathExpr(element=var_1)
    var_7 = [var_1]
    var_8 = 'IDENT'
    var_9 = 'div'
    var_10 = module_0.XPathExpr(element=var_9)
    var_11 = '1'
    var_12 = [var_11]
    var_13 = 'NUMBER'
    var_14 = module_0.XPathExpr(element=var_9)
    var_15 = [var_11]
    var_16 = module_0.XPathExpr(element=var_9)
    var_17 = [var_11]
    var_18 = 'contains'
    var_19 = 'class'



# Parsed testcases at query #109
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'gt'
    var_2 = 'NUMBER'
    var_3 = '0'
    var_4 = '2'
    var_5 = 'STRING'
    var_6 = '"hello"'
    var_7 = '1'



# Parsed testcases at query #110
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'test'
    var_12 = {var_10: var_11}



# Parsed testcases at query #111
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
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
    var_14 = ()
    var_15 = [var_6]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_11: var_18}
    var_20 = '//div'
    var_21 = 'Function'
    var_22 = ()
    var_23 = 'argument_types'
    var_24 = 'arguments'
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = 'Arg'
    var_29 = ()
    var_30 = 'value'
    var_31 = '1'
    var_32 = {var_30: var_31}
    var_33 = ()
    var_34 = 'IDENT'
    var_35 = [var_34]
    var_36 = lambda self: var_35
    var_37 = ()
    var_38 = 'test'
    var_39 = {var_30: var_38}



# Parsed testcases at query #112
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//div'
    var_2 = 'div'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'Function'
    var_5 = ()
    var_6 = 'argument_types'
    var_7 = 'arguments'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = lambda self: var_9
    var_11 = 'Argument'
    var_12 = ()
    var_13 = 'value'
    var_14 = '.bar'
    var_15 = {var_13: var_14}
    var_16 = 'descendant::*[contains(@class, "bar")]'
    var_17 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar ")]'
    var_18 = module_0.XPathExpr(var_1, var_2)
    var_19 = ()
    var_20 = [var_8]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = {var_13: var_2}
    var_24 = module_0.XPathExpr(var_1, var_2)
    var_25 = ()
    var_26 = [var_8]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '#myid'
    var_30 = {var_13: var_29}
    var_31 = '//div'
    var_32 = 'div'
    var_33 = module_0.XPathExpr(var_31, var_32)
    var_34 = 'Function'
    var_35 = ()
    var_36 = 'argument_types'
    var_37 = 'arguments'
    var_38 = 'NUMBER'
    var_39 = [var_38]
    var_40 = lambda self: var_39
    var_41 = 'Argument'
    var_42 = ()
    var_43 = 'value'
    var_44 = '1'
    var_45 = {var_43: var_44}
    var_46 = module_0.XPathExpr(var_31, var_32)
    var_47 = ()
    var_48 = 'IDENT'
    var_49 = [var_48]
    var_50 = lambda self: var_49
    var_51 = ()
    var_52 = 'test'
    var_53 = {var_43: var_52}



# Parsed testcases at query #113
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'content'
    var_19 = {var_10: var_18}
    var_20 = 'function'
    var_21 = ()
    var_22 = 'argument_types'
    var_23 = 'arguments'
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = 'arg'
    var_28 = ()
    var_29 = 'value'
    var_30 = '1'
    var_31 = {var_29: var_30}



# Parsed testcases at query #114
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = ':gt(0)'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3[var_4]
    var_6 = var_5.pseudo_class
    var_7 = var_0.xpath_gt_function(var_1, var_6)
    var_8 = module_0.XPathExpr()
    var_9 = ':gt(2)'
    var_10 = module_1.parse(var_9)
    var_11 = var_10[var_4]
    var_12 = var_11.pseudo_class
    var_13 = var_0.xpath_gt_function(var_8, var_12)
    var_14 = ':contains("text")'
    var_15 = module_1.parse(var_14)
    var_16 = var_15[var_4]
    var_17 = var_16.pseudo_class
    var_18 = module_0.XPathExpr()
    var_19 = var_0.xpath_gt_function(var_18, var_17)



# Parsed testcases at query #115
#--------------------------


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
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'test'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = [var_6]
    var_25 = lambda self: var_24
    var_26 = ()
    var_27 = "it's"
    var_28 = {var_11: var_27}
    var_29 = module_0.XPathExpr()
    var_30 = 'position() = 1'
    var_31 = var_29.add_post_condition(var_30)
    var_32 = ()
    var_33 = [var_6]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = {var_11: var_20}



# Parsed testcases at query #116
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = module_0.XPathExpr()
    var_2 = 0
    var_3 = module_0.XPathExpr()
    var_4 = 2
    var_5 = module_0.XPathExpr()
    var_6 = -1
    var_7 = module_0.XPathExpr()
    var_8 = 5
    var_9 = module_0.XPathExpr()
    var_10 = module_0.XPathExpr()
    var_11 = module_0.XPathExpr()
    var_12 = module_0.XPathExpr()
    var_13 = 1



# Parsed testcases at query #117
#--------------------------


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
    var_11 = '2'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '0'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}



# Parsed testcases at query #118
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()



# Parsed testcases at query #119
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'title'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'text'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '123'
    var_29 = {var_11: var_28}
    var_30 = module_0.XPathExpr()
    var_31 = ()
    var_32 = [var_6, var_6]
    var_33 = lambda self: var_32
    var_34 = ()
    var_35 = 'a'
    var_36 = {var_11: var_35}
    var_37 = ()
    var_38 = 'b'
    var_39 = {var_11: var_38}
    var_40 = module_0.XPathExpr()
    var_41 = ()
    var_42 = [var_6]
    var_43 = lambda self: var_42
    var_44 = ()
    var_45 = "it's"
    var_46 = {var_11: var_45}
    var_47 = module_0.XPathExpr()
    var_48 = ()
    var_49 = [var_6]
    var_50 = lambda self: var_49
    var_51 = ()
    var_52 = ''
    var_53 = {var_11: var_52}



# Parsed testcases at query #120
#--------------------------


import pyquery.cssselectpatch as module_0
import cssselect.parser as module_1

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'contains'
    var_2 = '"title"'
    var_3 = module_1.parse(var_2)
    var_4 = [var_3]
    var_5 = module_1.Function(var_1, var_4)
    var_6 = 'title'
    var_7 = module_1.parse(var_6)
    var_8 = [var_7]
    var_9 = module_1.Function(var_1, var_8)
    var_10 = '123'
    var_11 = module_1.parse(var_10)
    var_12 = [var_11]
    var_13 = module_1.Function(var_1, var_12)
    var_14 = var_0.xpath_contains_function(var_1, var_13)



# Parsed testcases at query #121
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = '.baz'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = 'div'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'IDENT'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'span'
    var_25 = {var_10: var_24}



# Parsed testcases at query #122
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = [var_6]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = {var_11: var_1}
    var_19 = ()
    var_20 = [var_6]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '.baz'
    var_24 = {var_11: var_23}
    var_25 = 'div'
    var_26 = 'Function'
    var_27 = ()
    var_28 = 'argument_types'
    var_29 = 'arguments'
    var_30 = 'NUMBER'
    var_31 = [var_30]
    var_32 = lambda self: var_31
    var_33 = 'Argument'
    var_34 = ()
    var_35 = 'value'
    var_36 = '0'
    var_37 = {var_35: var_36}



# Parsed testcases at query #123
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = ''
    var_3 = module_0.XPathExpr(var_1, var_1, var_2)
    var_4 = module_0.XPathExpr(var_1, var_1, var_2)
    var_5 = 'div'
    var_6 = ''
    var_7 = module_0.XPathExpr(var_5, var_5, var_6)



# Parsed testcases at query #124
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
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
    var_14 = 'title'
    var_15 = {var_13: var_14}
    var_16 = module_0.XPathExpr(var_1, var_2)
    var_17 = ()
    var_18 = 'IDENT'
    var_19 = [var_18]
    var_20 = lambda self: var_19
    var_21 = ()
    var_22 = {var_13: var_14}
    var_23 = module_0.XPathExpr(var_1, var_2)
    var_24 = ()
    var_25 = 'NUMBER'
    var_26 = [var_25]
    var_27 = lambda self: var_26
    var_28 = ()
    var_29 = '1'
    var_30 = {var_13: var_29}
    var_31 = module_0.XPathExpr(var_1, var_2)
    var_32 = ()
    var_33 = [var_8]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = "it's"
    var_37 = {var_13: var_36}



# Parsed testcases at query #125
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'Function'
    var_2 = ()
    var_3 = 'argument_types'
    var_4 = 'arguments'
    var_5 = 'STRING'
    var_6 = [var_5]
    var_7 = lambda self: var_6
    var_8 = 'Arg'
    var_9 = ()
    var_10 = 'value'
    var_11 = 'title'
    var_12 = {var_10: var_11}



# Parsed testcases at query #126
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = module_0.XPathExpr(var_1, var_2)
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
    var_14 = 'title'
    var_15 = {var_13: var_14}
    var_16 = '//div'
    var_17 = 'div'
    var_18 = module_0.XPathExpr(var_16, var_17)
    var_19 = ()
    var_20 = 'IDENT'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'content'
    var_25 = {var_13: var_24}
    var_26 = '//span'
    var_27 = 'span'
    var_28 = module_0.XPathExpr(var_26, var_27)
    var_29 = ()
    var_30 = 'NUMBER'
    var_31 = [var_30]
    var_32 = lambda self: var_31
    var_33 = ()
    var_34 = '123'
    var_35 = {var_13: var_34}
    var_36 = '//p'
    var_37 = 'p'
    var_38 = module_0.XPathExpr(var_36, var_37)
    var_39 = ()
    var_40 = []
    var_41 = lambda self: var_40
    var_42 = []
    var_43 = {var_6: var_41, var_7: var_42}
    var_44 = '//a'
    var_45 = 'a'
    var_46 = module_0.XPathExpr(var_44, var_45)
    var_47 = ()
    var_48 = [var_8]
    var_49 = lambda self: var_48
    var_50 = ()
    var_51 = "it's"
    var_52 = {var_13: var_51}



# Parsed testcases at query #127
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = ()
    var_15 = 'IDENT'
    var_16 = [var_15]
    var_17 = lambda self: var_16
    var_18 = ()
    var_19 = 'span'
    var_20 = {var_11: var_19}
    var_21 = 'div'
    var_22 = 'Function'
    var_23 = ()
    var_24 = 'argument_types'
    var_25 = 'arguments'
    var_26 = 'NUMBER'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = 'Argument'
    var_30 = ()
    var_31 = 'value'
    var_32 = '1'
    var_33 = {var_31: var_32}



# Parsed testcases at query #128
#--------------------------


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
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '3'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = 'STRING'
    var_21 = [var_20]
    var_22 = lambda self: var_21
    var_23 = ()
    var_24 = 'test'
    var_25 = {var_10: var_24}
    var_26 = ()
    var_27 = [var_5, var_5]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = '1'
    var_31 = {var_10: var_30}
    var_32 = ()
    var_33 = '2'
    var_34 = {var_10: var_33}



# Parsed testcases at query #129
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '/'
    var_2 = '*'



# Parsed testcases at query #130
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda : var_14
    var_16 = ()
    var_17 = '1'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda : var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = [var_5]
    var_27 = lambda : var_26
    var_28 = ()
    var_29 = '2'
    var_30 = {var_10: var_29}
    var_31 = ()
    var_32 = 'STRING'
    var_33 = [var_32]
    var_34 = lambda : var_33
    var_35 = ()
    var_36 = 'test'
    var_37 = {var_10: var_36}



# Parsed testcases at query #131
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = [var_1]
    var_3 = lambda : var_2
    var_4 = 'test text'
    var_5 = 'IDENT'
    var_6 = [var_5]
    var_7 = lambda : var_6
    var_8 = 'test_ident'
    var_9 = 'NUMBER'
    var_10 = [var_9]
    var_11 = lambda : var_10
    var_12 = 1



# Parsed testcases at query #132
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'lt'
    var_2 = 'NUMBER'
    var_3 = '2'
    var_4 = '0'
    var_5 = 'STRING'
    var_6 = 'invalid'
    var_7 = '1'



# Parsed testcases at query #133
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '//h1'
    var_2 = 'h1'
    var_3 = 'Function'
    var_4 = ()
    var_5 = 'argument_types'
    var_6 = 'arguments'
    var_7 = 'STRING'
    var_8 = [var_7]
    var_9 = lambda self: var_8
    var_10 = 'Arg'
    var_11 = ()
    var_12 = 'value'
    var_13 = 'title'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'content'
    var_21 = {var_12: var_20}
    var_22 = '//h1'
    var_23 = 'h1'
    var_24 = 'Function'
    var_25 = ()
    var_26 = 'argument_types'
    var_27 = 'arguments'
    var_28 = 'NUMBER'
    var_29 = [var_28]
    var_30 = lambda self: var_29
    var_31 = 'Arg'
    var_32 = ()
    var_33 = 'value'
    var_34 = '1'
    var_35 = {var_33: var_34}



# Parsed testcases at query #134
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = '0'
    var_4 = '-1'
    var_5 = 'STRING'
    var_6 = 'test'



# Parsed testcases at query #135
#--------------------------


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
    var_12 = '.bar'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'div'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '1'
    var_29 = {var_11: var_28}
    var_30 = module_0.XPathExpr()
    var_31 = ()
    var_32 = [var_6]
    var_33 = lambda self: var_32
    var_34 = ()
    var_35 = '.foo > .bar'
    var_36 = {var_11: var_35}



# Parsed testcases at query #136
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '2'
    var_3 = 'STRING'



# Parsed testcases at query #137
#--------------------------


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
    var_9 = 'Argument'
    var_10 = ()
    var_11 = 'value'
    var_12 = 'test text'
    var_13 = {var_11: var_12}
    var_14 = module_0.XPathExpr()
    var_15 = ()
    var_16 = 'IDENT'
    var_17 = [var_16]
    var_18 = lambda self: var_17
    var_19 = ()
    var_20 = 'test_ident'
    var_21 = {var_11: var_20}
    var_22 = module_0.XPathExpr()
    var_23 = ()
    var_24 = 'NUMBER'
    var_25 = [var_24]
    var_26 = lambda self: var_25
    var_27 = ()
    var_28 = '42'
    var_29 = {var_11: var_28}



# Parsed testcases at query #138
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '1'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '5'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}



# Parsed testcases at query #139
#--------------------------


import pyquery.cssselectpatch as module_0

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
    var_13 = ()
    var_14 = 'IDENT'
    var_15 = [var_14]
    var_16 = lambda self: var_15
    var_17 = ()
    var_18 = 'div'
    var_19 = {var_10: var_18}
    var_20 = ()
    var_21 = 'NUMBER'
    var_22 = [var_21]
    var_23 = lambda self: var_22
    var_24 = ()
    var_25 = '1'
    var_26 = {var_10: var_25}
    var_27 = ()
    var_28 = [var_5]
    var_29 = lambda self: var_28
    var_30 = ()
    var_31 = '.test'
    var_32 = {var_10: var_31}



# Parsed testcases at query #140
#--------------------------


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
    var_8 = 'Argument'
    var_9 = ()
    var_10 = 'value'
    var_11 = '0'
    var_12 = {var_10: var_11}
    var_13 = ()
    var_14 = [var_5]
    var_15 = lambda self: var_14
    var_16 = ()
    var_17 = '2'
    var_18 = {var_10: var_17}
    var_19 = ()
    var_20 = [var_5]
    var_21 = lambda self: var_20
    var_22 = ()
    var_23 = '-1'
    var_24 = {var_10: var_23}
    var_25 = ()
    var_26 = 'STRING'
    var_27 = [var_26]
    var_28 = lambda self: var_27
    var_29 = ()
    var_30 = 'test'
    var_31 = {var_10: var_30}
    var_32 = ()
    var_33 = [var_5, var_5]
    var_34 = lambda self: var_33
    var_35 = ()
    var_36 = '1'
    var_37 = {var_10: var_36}



