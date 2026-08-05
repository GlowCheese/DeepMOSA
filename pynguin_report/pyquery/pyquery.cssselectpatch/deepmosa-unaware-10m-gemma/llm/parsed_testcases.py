####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = str(var_2)



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = '"0"'
    var_7 = 'p'
    var_8 = module_0.XPathExpr(var_7)
    var_9 = str(var_8)



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = 'STRING'
    var_5 = 'abc'
    var_6 = 'p'
    var_7 = module_0.XPathExpr(var_6)
    var_8 = str(var_7)
    assert var_8 == 'p[position() < 1]'



# Parsed testcases at query #4
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = 'Expected a single integer for :gt(), got'
    var_7 = 'Expected a single integer for :lt(), got'
    var_8 = 'p'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = str(var_9)



# Parsed testcases at query #5
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'
    var_5 = 'abc'



# Parsed testcases at query #6
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'STRING'
    var_4 = 'abc'
    var_5 = 'position() > 3'



# Parsed testcases at query #7
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "contains(., 'target_text')"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = module_0.XPathExpr(var_1)
    var_8 = str(var_7)
    var_9 = 'NUMBER'
    var_10 = module_0.XPathExpr(var_1)
    var_11 = 'BOOLEAN'



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = '1'
    var_5 = str(var_2)
    var_6 = 'STRING'
    var_7 = 'abc'
    var_8 = 'Expected a single integer for :gt(), got'
    var_9 = 'Expected a single integer'
    var_10 = '0'
    var_11 = 'div'
    var_12 = module_0.XPathExpr(var_11)
    var_13 = str(var_12)



# Parsed testcases at query #9
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 2'
    var_3 = 'STRING'
    var_4 = 'position() < 6'



# Parsed testcases at query #10
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = '.bar'
    var_5 = 'descendant::*[@class="bar"]'
    var_6 = str(var_3)
    var_7 = module_0.XPathExpr(var_1, var_2)
    var_8 = 'descendant::div'
    var_9 = str(var_7)
    var_10 = 'NUMBER'
    var_11 = 'BOOLEAN'



# Parsed testcases at query #11
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = lambda selector, prefix='': f'{prefix}{selector}'
    var_2 = '.bar'
    var_3 = 'STRING'
    var_4 = 'descendant::.bar'
    var_5 = 'div'
    var_6 = 'IDENT'
    var_7 = 'descendant::div'
    var_8 = '123'
    var_9 = 'NUMBER'



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'STRING'
    var_5 = 'descendant::*[contains(concat("class=", @class), " bar ")]'
    var_6 = str(var_3)
    var_7 = 'NUMBER'
    var_8 = 'IDENT'
    var_9 = 'descendant::div'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = 'position() = 1'
    var_5 = [var_1]
    var_6 = '5'
    var_7 = 'position() = 6'
    var_8 = 'STRING'
    var_9 = [var_8]
    var_10 = '"0"'
    var_11 = 'IDENT'
    var_12 = [var_11]
    var_13 = 'foo'



# Parsed testcases at query #16
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = '.bar'
    var_3 = 'descendant::*[@class="bar"]'
    var_4 = 'IDENT'
    var_5 = 'div'
    var_6 = 'descendant::div'
    var_7 = 'NUMBER'
    var_8 = '123'



# Parsed testcases at query #17
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = 'foo'
    var_7 = 'p'
    var_8 = module_0.XPathExpr(var_7)
    var_9 = str(var_8)



# Parsed testcases at query #18
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = 'abc'
    var_7 = 'p'
    var_8 = module_0.XPathExpr(var_7)
    var_9 = str(var_8)



# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = 'abc'
    var_7 = 'p'
    var_8 = module_0.XPathExpr(var_7)
    var_9 = str(var_8)



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 2'
    var_3 = 'STRING'
    var_4 = 'position() = 1'



# Parsed testcases at query #21
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 2'
    var_3 = 'STRING'
    var_4 = 'foo'
    var_5 = 'Expected a single integer for :gt(), got'
    var_6 = 'Expected a single integer for :lt(), got'
    var_7 = 'position() < 1'



# Parsed testcases at query #22
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = '0'
    var_4 = 'NUMBER'
    var_5 = [var_4]
    var_6 = str(var_2)
    var_7 = '5'
    var_8 = [var_4]
    var_9 = str(var_2)
    var_10 = 'abc'
    var_11 = 'STRING'
    var_12 = [var_11]
    var_13 = 'some_id'
    var_14 = 'IDENT'
    var_15 = [var_14]



# Parsed testcases at query #23
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'position() > 6'
    var_4 = 'STRING'
    var_5 = 'IDENT'



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = 'STRING'
    var_5 = 'div'
    var_6 = module_0.XPathExpr(var_5)
    var_7 = module_0.XPathExpr(var_5)
    var_8 = str(var_7)



# Parsed testcases at query #25
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'hello'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = 'title'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = module_0.XPathExpr()
    var_12 = 'LIST'
    var_13 = module_0.XPathExpr()



# Parsed testcases at query #26
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'position() > 6'
    var_4 = 'STRING'



# Parsed testcases at query #27
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::*[@class="bar"]'
    var_3 = 'NUMBER'
    var_4 = 'IDENT'
    var_5 = 'descendant::div'
    var_6 = 'BOOLEAN'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = var_0.xpath_last_pseudo(var_2)



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::.baz'
    var_3 = 'IDENT'
    var_4 = 'descendant::div'
    var_5 = 'NUMBER'
    var_6 = 1



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = '5'
    var_4 = 'position() > 6'
    var_5 = 'STRING'
    var_6 = '"abc"'



# Parsed testcases at query #4
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = var_0.xpath_first_pseudo(var_2)
    var_4 = str(var_2)
    assert var_4 == 'p[position() = 1]'

import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'class="container"'
    var_4 = var_2.add_post_condition(var_3)
    var_5 = var_0.xpath_first_pseudo(var_2)
    var_6 = str(var_2)



# Parsed testcases at query #5
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'input'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = var_0.xpath_password_pseudo(var_2)



# Parsed testcases at query #6
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'
    var_5 = 'IDENT'



# Parsed testcases at query #7
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'hello'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = 'some_ident'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = 'div'
    var_12 = module_0.XPathExpr(var_11)
    var_13 = 'LIST'
    var_14 = 'div'
    var_15 = module_0.XPathExpr(var_14)



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 2'
    var_3 = 'STRING'
    var_4 = 'Expected a single integer for :gt()'
    var_5 = 'Expected a single integer for :lt()'
    var_6 = 'position() < 1'



# Parsed testcases at query #9
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::.baz'
    var_3 = 'IDENT'
    var_4 = 'NUMBER'
    var_5 = 'descendant::div'



# Parsed testcases at query #10
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = "'target_text'"
    var_3 = "contains(., 'target_text')"
    var_4 = 'NUMBER'
    var_5 = 'IDENT'
    var_6 = '"target_text"'
    var_7 = 'contains(., "target_text")'
    var_8 = 'BOOLEAN'



# Parsed testcases at query #11
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'STRING'
    var_5 = 'descendant::*.bar'
    var_6 = str(var_3)
    var_7 = 'IDENT'
    var_8 = module_0.XPathExpr(var_2, var_2)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = module_0.XPathExpr()
    var_12 = module_0.XPathExpr()



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'hello'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = 'myident'
    var_8 = 'p'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = str(var_9)
    var_11 = 'NUMBER'
    var_12 = 'span'
    var_13 = module_0.XPathExpr(var_12)



# Parsed testcases at query #13
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::*[@class="bar"]'
    var_3 = 'IDENT'
    var_4 = 'descendant::div'
    var_5 = 'NUMBER'



# Parsed testcases at query #14
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'p'
    var_6 = module_0.XPathExpr(var_5)
    var_7 = str(var_6)
    var_8 = 'STRING'



# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'STRING'
    var_4 = 'position() > 6'



# Parsed testcases at query #16
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::*[@class="baz"]'
    var_3 = 'IDENT'
    var_4 = 'descendant::div'
    var_5 = 'NUMBER'



# Parsed testcases at query #17
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'STRING'
    var_5 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar") and (not(@class) or substring-before(substring-after(concat(" ", @class, " "), " "), " ") = "bar" or substring-after(concat(" ", @class, " "), " ") = "bar")]'
    var_6 = 'descendant::*.bar'
    var_7 = str(var_3)
    var_8 = 'IDENT'
    var_9 = module_0.XPathExpr(var_1, var_2)
    var_10 = str(var_9)
    var_11 = 'NUMBER'
    var_12 = module_0.XPathExpr()
    var_13 = 'BOOLEAN'
    var_14 = module_0.XPathExpr()



# Parsed testcases at query #18
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 3'
    var_3 = 'STRING'
    var_4 = 'not_a_number'
    var_5 = 'position() = 1'



# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = 'descendant::*.bar'
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = module_0.XPathExpr(var_1)
    var_8 = str(var_7)
    var_9 = 'NUMBER'
    var_10 = module_0.XPathExpr(var_1)
    var_11 = module_0.XPathExpr(var_1)
    var_12 = str(var_11)



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'STRING'
    var_5 = 'descendant::*[@class="bar"]'
    var_6 = str(var_3)
    var_7 = 'NUMBER'
    var_8 = 'IDENT'



# Parsed testcases at query #21
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::*[@class="baz"]'
    var_3 = 'IDENT'
    var_4 = 'descendant::div'
    var_5 = 'NUMBER'



# Parsed testcases at query #22
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = module_0.XPathExpr(var_1)
    var_7 = '0'
    var_8 = str(var_6)



# Parsed testcases at query #23
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = module_0.XPathExpr(var_1)
    var_6 = str(var_5)
    var_7 = module_0.XPathExpr(var_1)
    var_8 = 'STRING'
    var_9 = 'Expected a single integer for :gt()'
    var_10 = 'Expected a single integer for :lt()'



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'test'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = "'some_id'"
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = 'div'
    var_12 = module_0.XPathExpr(var_11)
    var_13 = 'div'
    var_14 = module_0.XPathExpr(var_13)



# Parsed testcases at query #25
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'test'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = 'some_id'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = 'div'
    var_12 = module_0.XPathExpr(var_11)
    var_13 = 'div'
    var_14 = module_0.XPathExpr(var_13)



# Parsed testcases at query #26
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = str(var_2)
    var_5 = 'IDENT'
    var_6 = module_0.XPathExpr(var_1)
    var_7 = str(var_6)
    var_8 = 'NUMBER'
    var_9 = module_0.XPathExpr(var_1)



# Parsed testcases at query #27
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'position() > 6'
    var_4 = 'STRING'
    var_5 = 'abc'



# Parsed testcases at query #28
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'p'
    var_6 = module_0.XPathExpr(var_5)
    var_7 = str(var_6)
    var_8 = 'STRING'
    var_9 = 'IDENT'



# Parsed testcases at query #29
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'title'"
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = module_0.XPathExpr(var_1)
    var_8 = str(var_7)
    var_9 = 'NUMBER'
    var_10 = module_0.XPathExpr()
    var_11 = 'BOOLEAN'
    var_12 = module_0.XPathExpr()



