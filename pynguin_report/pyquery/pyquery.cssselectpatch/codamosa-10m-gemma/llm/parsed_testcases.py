####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_8 = 'span'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = 'STRING'
    var_11 = 'IDENT'



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = '.bar'
    var_5 = 'STRING'
    var_6 = 'descendant::*.bar'
    var_7 = str(var_3)
    var_8 = 'IDENT'
    var_9 = 'descendant::div'
    var_10 = module_0.XPathExpr(var_2, var_2)
    var_11 = str(var_10)
    var_12 = '123'
    var_13 = 'NUMBER'
    var_14 = module_0.XPathExpr()
    var_15 = 'true'
    var_16 = 'BOOLEAN'
    var_17 = module_0.XPathExpr()



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = "'test-text'"
    var_3 = "contains(., 'test-text')"
    var_4 = 'IDENT'
    var_5 = 'some_id'
    var_6 = 'contains(., some_id)'
    var_7 = 'NUMBER'
    var_8 = 'BOOLEAN'



# Parsed testcases at query #4
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'



# Parsed testcases at query #5
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'



# Parsed testcases at query #6
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
    var_9 = 'not_a_number'
    var_10 = 'IDENT'
    var_11 = 'foo'



# Parsed testcases at query #7
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '0'
    var_3 = [var_1]
    var_4 = 'position() = 1'
    var_5 = '5'
    var_6 = [var_1]
    var_7 = 'position() = 6'
    var_8 = 'STRING'
    var_9 = 'abc'
    var_10 = [var_8]



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = '[position() > 2]'
    var_5 = 'STRING'
    var_6 = 'div'
    var_7 = module_0.XPathExpr(var_6)
    var_8 = module_0.XPathExpr(var_6)
    var_9 = str(var_8)



# Parsed testcases at query #9
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
    var_10 = '"text"'
    var_11 = 'IDENT'
    var_12 = [var_11]
    var_13 = 'some_id'



# Parsed testcases at query #10
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 2'
    var_3 = 'STRING'
    var_4 = 'position() > 1'



# Parsed testcases at query #11
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = "'target'"
    var_5 = str(var_2)
    var_6 = 'NUMBER'
    var_7 = 'IDENT'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = module_0.XPathExpr(var_1)
    var_6 = str(var_5)
    var_7 = 'STRING'
    var_8 = 'foo'



# Parsed testcases at query #13
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = '1'
    var_5 = str(var_2)
    var_6 = 'STRING'
    var_7 = '"text"'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = '0'
    var_10 = str(var_8)



# Parsed testcases at query #14
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
    var_7 = 'some_id'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = module_0.XPathExpr()
    var_12 = 'BOOLEAN'
    var_13 = module_0.XPathExpr()



# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'



# Parsed testcases at query #16
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
    var_8 = 'h1'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = str(var_9)
    var_11 = 'NUMBER'
    var_12 = 'div'
    var_13 = module_0.XPathExpr(var_12)
    var_14 = 'LIST'
    var_15 = 'div'
    var_16 = module_0.XPathExpr(var_15)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_9 = 'span'
    var_10 = module_0.XPathExpr(var_9)



# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::*[@class="bar"]'
    var_3 = 'IDENT'
    var_4 = 'descendant::div'
    var_5 = 'NUMBER'



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = '0'
    var_5 = str(var_2)
    var_6 = 'STRING'
    var_7 = 'not_a_number'
    var_8 = 'p'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = '5'
    var_11 = str(var_9)



# Parsed testcases at query #21
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = '0'
    var_5 = str(var_2)
    var_6 = '5'
    var_7 = str(var_2)
    var_8 = 'STRING'
    var_9 = '"0"'
    var_10 = 'IDENT'
    var_11 = 'foo'



# Parsed testcases at query #22
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
    var_9 = 'abc'



# Parsed testcases at query #23
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
    var_9 = module_0.XPathExpr()
    var_10 = module_0.XPathExpr()



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '.bar'
    var_2 = 'STRING'
    var_3 = 'descendant::*[@class="bar"]'
    var_4 = 'div'
    var_5 = 'IDENT'
    var_6 = 'descendant::div'
    var_7 = '123'
    var_8 = 'NUMBER'
    var_9 = 'true'
    var_10 = 'BOOLEAN'



# Parsed testcases at query #25
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = module_0.XPathExpr(var_1)
    var_6 = str(var_5)
    var_7 = 'STRING'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = 'descendant::*.bar'
    var_5 = str(var_2)
    var_6 = 'NUMBER'
    var_7 = 'IDENT'



# Parsed testcases at query #2
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = var_0.xpath_first_pseudo(var_2)
    var_4 = str(var_2)
    assert var_4 == 'p[position() = 1]'



# Parsed testcases at query #3
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = "@type = 'image' and name(.) = 'input'"



# Parsed testcases at query #4
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
    var_9 = 'span'
    var_10 = module_0.XPathExpr(var_9)
    var_11 = str(var_10)



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
    var_2 = '1'
    var_3 = 'position() < 2'
    var_4 = '0'
    var_5 = 'position() < 1'
    var_6 = 'STRING'
    var_7 = '"text"'
    var_8 = 'Expected a single integer for :gt()'
    var_9 = 'gt'



# Parsed testcases at query #7
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() < 2'
    var_3 = 'STRING'
    var_4 = 'position() < 1'



# Parsed testcases at query #8
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = '"target text"'
    var_5 = lambda x: x
    var_6 = str(var_2)
    var_7 = 'IDENT'
    var_8 = 'some_id'
    var_9 = module_0.XPathExpr(var_1)
    var_10 = str(var_9)
    var_11 = 'NUMBER'
    var_12 = '123'
    var_13 = module_0.XPathExpr(var_1)
    var_14 = 'BOOLEAN'
    var_15 = module_0.XPathExpr(var_1)



# Parsed testcases at query #9
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = 'hello'
    var_5 = "'hello'"
    var_6 = str(var_2)
    var_7 = 'IDENT'
    var_8 = 'title'
    var_9 = module_0.XPathExpr(var_1)
    var_10 = str(var_9)
    var_11 = 'NUMBER'
    var_12 = 123
    var_13 = 'div'
    var_14 = module_0.XPathExpr(var_13)
    var_15 = 'LIST'
    var_16 = 'div'
    var_17 = module_0.XPathExpr(var_16)



# Parsed testcases at query #10
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'STRING'
    var_4 = 'position() = 6'



# Parsed testcases at query #11
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'



# Parsed testcases at query #12
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '1'
    var_3 = 'position() < 2'
    var_4 = '0'
    var_5 = 'position() < 1'
    var_6 = 'STRING'
    var_7 = 'abc'



# Parsed testcases at query #13
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = module_0.XPathExpr(var_1)
    var_5 = 'STRING'
    var_6 = 'foo'
    var_7 = module_0.XPathExpr(var_1)



# Parsed testcases at query #14
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = "'title'"
    var_3 = "contains(., 'title')"
    var_4 = 'IDENT'
    var_5 = 'some_id'
    var_6 = 'contains(., some_id)'
    var_7 = 'NUMBER'
    var_8 = 'BOOLEAN'



# Parsed testcases at query #15
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() = 1'
    var_3 = 'position() = 6'
    var_4 = 'STRING'
    var_5 = 'IDENT'



# Parsed testcases at query #16
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = '1'
    var_3 = 'position() < 2'
    var_4 = 'STRING'
    var_5 = 'abc'



# Parsed testcases at query #17
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = '"test"'
    var_5 = str(var_2)
    var_6 = module_0.XPathExpr(var_1)
    var_7 = 'IDENT'
    var_8 = 'some_id'
    var_9 = str(var_6)
    var_10 = 'NUMBER'
    var_11 = '123'



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'STRING'
    var_2 = 'descendant::.bar'
    var_3 = 'IDENT'
    var_4 = 'NUMBER'
    var_5 = 'descendant::div'



# Parsed testcases at query #20
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = module_0.XPathExpr(var_1)
    var_6 = 'STRING'
    var_7 = module_0.XPathExpr(var_1)
    var_8 = str(var_7)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'p'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'NUMBER'
    var_4 = str(var_2)
    var_5 = 'STRING'
    var_6 = 'Expected a single integer for :gt(), got'
    var_7 = 'Expected a single integer for :lt(), got'
    var_8 = 'div'
    var_9 = module_0.XPathExpr(var_8)
    var_10 = str(var_9)



# Parsed testcases at query #23
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = '*'
    var_3 = module_0.XPathExpr(var_1, var_2)
    var_4 = 'STRING'
    var_5 = 'descendant::*[contains(concat(" ", normalize-space(@class), " "), " bar")]'
    var_6 = str(var_3)
    var_7 = 'IDENT'
    var_8 = module_0.XPathExpr(var_1, var_2)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = module_0.XPathExpr()
    var_12 = module_0.XPathExpr()



# Parsed testcases at query #24
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = lambda selector, prefix='': f'{prefix}{selector}'
    var_2 = 'STRING'
    var_3 = '.bar'
    var_4 = 'descendant::.bar'
    var_5 = 'IDENT'
    var_6 = 'div'
    var_7 = 'descendant::div'
    var_8 = 'NUMBER'
    var_9 = '123'
    var_10 = 'BOOLEAN'
    var_11 = 'true'



# Parsed testcases at query #25
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
    var_7 = 'contains(., some_id)'
    var_8 = str(var_6)
    var_9 = var_7 in var_8
    var_10 = "contains(., 'some_id')"
    var_11 = str(var_6)
    var_12 = var_10 in var_11
    var_13 = 'NUMBER'
    var_14 = module_0.XPathExpr()
    var_15 = 'LIST'
    var_16 = module_0.XPathExpr()



# Parsed testcases at query #26
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
    var_9 = module_0.XPathExpr()



# Parsed testcases at query #27
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = '"div"'
    var_2 = 'STRING'
    var_3 = [var_2]
    var_4 = 'descendant::div'
    var_5 = 'div'
    var_6 = 'IDENT'
    var_7 = [var_6]
    var_8 = 123
    var_9 = 'NUMBER'
    var_10 = [var_9]



# Parsed testcases at query #28
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'NUMBER'
    var_2 = 'position() > 1'
    var_3 = 'position() > 6'
    var_4 = 'STRING'



# Parsed testcases at query #29
#--------------------------


import pyquery.cssselectpatch as module_0

def test_case_0():
    var_0 = module_0.JQueryTranslator()
    var_1 = 'div'
    var_2 = module_0.XPathExpr(var_1)
    var_3 = 'STRING'
    var_4 = '"hello"'
    var_5 = str(var_2)
    var_6 = 'IDENT'
    var_7 = 'some_id'
    var_8 = module_0.XPathExpr(var_1)
    var_9 = str(var_8)
    var_10 = 'NUMBER'
    var_11 = '123'
    var_12 = module_0.XPathExpr()
    var_13 = 'BOOLEAN'
    var_14 = module_0.XPathExpr()



# Parsed testcases at query #30
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



