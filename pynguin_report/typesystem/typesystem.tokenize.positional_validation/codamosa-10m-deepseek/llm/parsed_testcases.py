####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = e.messages()[0]
    var_7 = 'valid'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'inner'
    var_15 = 'Bob'
    var_16 = {var_0: var_15}
    var_17 = {var_14: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = error.messages()[0]



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = (var_1, var_1)
    var_3 = (var_1, var_1)
    var_4 = module_0.Token(var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = 'age'
    var_10 = 'not an int'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 'All tests passed.'
    var_16 = print(var_15)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 1
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = module_0.Token(var_8)
    var_10 = 123
    var_11 = 'thirty'
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = module_0.Token(var_12)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'nested'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = error.messages()[0]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'thirty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = 5
    var_10 = module_0.Token(var_8)
    var_11 = error.messages()[0]
    var_12 = 'thirty'
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = error.messages()[0]
    var_17 = 123
    var_18 = {var_0: var_17, var_1: var_12}
    var_19 = 20
    var_20 = module_0.Token(var_18)
    var_21 = sorted(error.messages(), key=lambda m: m.text)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = None
    var_5 = 10
    var_6 = 15



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 0
    var_7 = {var_5: var_6}
    var_8 = 20
    var_9 = {var_5: var_8}
    var_10 = module_0.Token(var_4)
    var_11 = module_1.validate_with_positions(token=var_10, validator=var_0)
    var_12 = {}
    var_13 = {var_5: var_6}
    var_14 = {var_5: var_6}
    var_15 = module_0.Token(var_12)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 'nested'
    var_13 = 'address'
    var_14 = '123 Main St'
    var_15 = {var_13: var_14}
    var_16 = {var_0: var_2, var_12: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {}
    var_19 = {var_0: var_2, var_12: var_18}
    var_20 = module_0.Token(var_19)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'Bob'
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'Charlie'
    var_11 = 'thirty'
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 'info'
    var_16 = 'Dave'
    var_17 = 40
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = {var_15: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 'Eve'
    var_22 = {var_0: var_21}
    var_23 = {var_15: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = error.messages()[0]



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'not_an_int'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_3}
    var_13 = module_0.Token(var_12)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = e.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = e.messages()[0]



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = {var_1: var_9}
    var_13 = module_0.Token(var_12)
    var_14 = 'All tests passed successfully.'
    var_15 = print(var_14)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = True
    var_7 = module_1.Field()
    var_8 = module_2.validate_with_positions(token=var_5, validator=var_7)
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = module_2.validate_with_positions(token=var_10, validator=var_7)
    var_12 = ''
    var_13 = {var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = module_1.Field()
    var_16 = module_2.validate_with_positions(token=var_14, validator=var_15)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = (var_5, var_5)
    var_7 = 1
    var_8 = (var_7, var_5)
    var_9 = ''
    var_10 = module_0.Token(var_4, var_9)
    var_11 = {var_0: var_2}
    var_12 = (var_5, var_5)
    var_13 = (var_7, var_5)
    var_14 = module_0.Token(var_11, var_9)
    var_15 = error.messages()[0]
    var_16 = 'not an int'
    var_17 = {var_0: var_2, var_1: var_16}
    var_18 = (var_5, var_5)
    var_19 = (var_7, var_5)
    var_20 = module_0.Token(var_17, var_9)
    var_21 = error.messages()[0]



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = 'Bob'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = error.messages()[0]
    var_11 = 'Charlie'
    var_12 = 'thirty'
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = error.messages()[0]
    var_16 = 'nested'
    var_17 = 'Dave'
    var_18 = {}
    var_19 = {var_0: var_17, var_16: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = error.messages()[0]
    var_22 = 'All tests passed.'
    var_23 = print(var_22)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_1: var_3}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 123
    var_13 = {var_0: var_12, var_1: var_9}
    var_14 = module_0.Token(var_13)
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'nested'
    var_15 = 'address'
    var_16 = '123 Main St'
    var_17 = {var_15: var_16}
    var_18 = {var_0: var_2, var_14: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = {}
    var_21 = {var_0: var_2, var_14: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = error.messages()[0]



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = error.messages()[0]
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = error.messages()[0]
    var_13 = 123
    var_14 = {var_0: var_13, var_1: var_9}
    var_15 = module_0.Token(var_14)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 'line_index'
    var_7 = 'column_index'
    var_8 = 0
    var_9 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_10 = 20
    var_11 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_12 = module_0.Token(var_4)
    var_13 = {var_0: var_2}
    var_14 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_15 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_16 = module_0.Token(var_13)
    var_17 = error.messages()[0]



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'nested'
    var_15 = 'address'
    var_16 = '123 Main St'
    var_17 = {var_15: var_16}
    var_18 = {var_0: var_2, var_14: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = {}
    var_21 = {var_0: var_2, var_14: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = error.messages()[0]



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'char_index'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 10
    var_7 = {var_3: var_6}
    var_8 = module_0.Token(var_2)
    var_9 = 'age'
    var_10 = 'not an integer'
    var_11 = {var_0: var_1, var_9: var_10}
    var_12 = {var_3: var_4}
    var_13 = 20
    var_14 = {var_3: var_13}
    var_15 = module_0.Token(var_11)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 5
    var_3 = 'invalid'
    var_4 = 7
    var_5 = None
    var_6 = 4



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'thirty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'user'
    var_15 = {var_0: var_11, var_1: var_8}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = 'nested'
    var_19 = {var_0: var_11, var_1: var_8}
    var_20 = {var_14: var_19}
    var_21 = {var_18: var_20}
    var_22 = module_0.Token(var_21)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 'line_index'
    var_7 = 'column_index'
    var_8 = 0
    var_9 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_10 = 10
    var_11 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_12 = module_0.Token(var_4)
    var_13 = {var_0: var_2}
    var_14 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_15 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_16 = module_0.Token(var_13)
    var_17 = error.messages()[0]



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'char_index'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = 10
    var_7 = {var_3: var_6}
    var_8 = module_0.Token(var_2)
    var_9 = {}
    var_10 = {var_3: var_4}
    var_11 = {var_3: var_6}
    var_12 = module_0.Token(var_9)
    var_13 = error.messages()[0]
    var_14 = 123
    var_15 = {var_0: var_14}
    var_16 = {var_3: var_4}
    var_17 = {var_3: var_6}
    var_18 = module_0.Token(var_15)
    var_19 = error.messages()[0]



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = error.messages()[0]
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = error.messages()[0]



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = error.messages()[0]
    var_8 = 'invalid'
    var_9 = module_0.Token(var_8)
    var_10 = module_1.validate_with_positions(token=var_9, validator=var_0)
    var_11 = error.messages()[0]



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'invalid'
    var_8 = module_0.Token(var_7)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 'invalid'
    var_2 = 0
    var_3 = 10
    var_4 = None
    var_5 = 5
    var_6 = 15
    var_7 = 'All test cases passed successfully.'
    var_8 = print(var_7)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'info'
    var_15 = {var_0: var_2, var_1: var_10}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'thirty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 0
    var_5 = (var_3, var_4)
    var_6 = 10
    var_7 = (var_3, var_6)
    var_8 = module_0.Token(var_2)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'test'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'invalid'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'nested'
    var_15 = 'nested_field'
    var_16 = {var_15: var_3}
    var_17 = {var_14: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = error.messages()[0]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 10
    var_3 = e.messages()[0]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'age'
    var_8 = 'not an integer'
    var_9 = {var_7: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 'info'
    var_12 = 'Jane'
    var_13 = {var_0: var_12}
    var_14 = {var_11: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = {}
    var_17 = {var_11: var_16}
    var_18 = module_0.Token(var_17)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = 'age'
    var_6 = 'not an integer'
    var_7 = {var_0: var_1, var_5: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = 25
    var_10 = {var_0: var_1, var_5: var_9}
    var_11 = module_0.Token(var_10)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'thirty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 'All test cases passed!'
    var_12 = print(var_11)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 'invalid'
    var_2 = None
    var_3 = 'All test cases passed.'
    var_4 = print(var_3)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 'nested'
    var_13 = 'address'
    var_14 = '123 Main St'
    var_15 = {var_13: var_14}
    var_16 = {var_0: var_2, var_12: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {}
    var_19 = {var_0: var_2, var_12: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 123
    var_22 = {var_13: var_21}
    var_23 = {var_0: var_2, var_12: var_22}
    var_24 = module_0.Token(var_23)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = module_0.Token(var_8)
    var_10 = True
    var_11 = error.messages()[0]
    var_12 = 'thirty'
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = error.messages()[0]
    var_16 = 'All test cases passed!'
    var_17 = print(var_16)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_1: var_3}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'invalid'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 'invalid'
    var_2 = 'existing_field'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'multiple_errors'
    var_6 = 'All test cases passed.'
    var_7 = print(var_6)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = module_0.Token(var_4)
    var_8 = 'AliceAliceAlice'
    var_9 = {var_0: var_8, var_1: var_3}
    var_10 = module_0.Token(var_9)
    var_11 = {var_1: var_3}
    var_12 = module_0.Token(var_11)
    var_13 = -5
    var_14 = {var_0: var_2, var_1: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = 'All tests passed successfully!'
    var_17 = print(var_16)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'address'
    var_15 = 'street'
    var_16 = 'city'
    var_17 = '123 Main St'
    var_18 = 'Springfield'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = {var_0: var_2, var_14: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = {var_15: var_17}
    var_23 = {var_0: var_2, var_14: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = error.messages()[0]



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = (var_3, var_3)
    var_5 = (var_3, var_0)
    var_6 = module_1.Token(var_2)
    var_7 = module_2.validate_with_positions(token=var_6, validator=var_1)
    assert var_7 == 'hello'
    var_8 = True
    var_9 = module_0.Integer()
    var_10 = None
    var_11 = (var_3, var_3)
    var_12 = (var_3, var_3)
    var_13 = module_1.Token(var_10)
    var_14 = module_2.validate_with_positions(token=var_13, validator=var_9)
    var_15 = error.messages()[0]
    var_16 = module_0.Integer()
    var_17 = 'name'
    var_18 = 'age'
    var_19 = 'Alice'
    var_20 = 30
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = (var_3, var_3)
    var_23 = 20
    var_24 = (var_3, var_23)
    var_25 = (var_17,)
    var_26 = (var_17,)
    var_27 = 8
    var_28 = (var_3, var_27)
    var_29 = 16
    var_30 = (var_3, var_29)
    var_31 = (var_17,)
    var_32 = 13
    var_33 = (var_3, var_32)
    var_34 = 18
    var_35 = (var_3, var_34)
    var_36 = lambda index: Token(value=var_19 if index == var_25 else var_20, start=var_28 if index == var_26 else var_30, end=var_33 if index == var_31 else var_35)
    var_37 = module_1.Token(var_21)
    var_38 = 'TooLongName'
    var_39 = 'not_an_integer'
    var_40 = {var_17: var_38, var_18: var_39}
    var_41 = (var_3, var_3)
    var_42 = (var_3, var_20)
    var_43 = (var_17,)
    var_44 = (var_17,)
    var_45 = (var_3, var_27)
    var_46 = (var_3, var_29)
    var_47 = (var_17,)
    var_48 = 19
    var_49 = (var_3, var_48)
    var_50 = 29
    var_51 = (var_3, var_50)
    var_52 = lambda index: Token(value=var_38 if index == var_43 else var_39, start=var_45 if index == var_44 else var_46, end=var_49 if index == var_47 else var_51)
    var_53 = module_1.Token(var_40)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 5
    var_6 = 9
    var_7 = None
    var_8 = lambda _: var_7
    var_9 = module_0.Token(var_1)
    var_10 = lambda index: var_9
    var_11 = module_0.Token(var_2)
    var_12 = {}
    var_13 = lambda _: var_7
    var_14 = module_0.Token(var_7)
    var_15 = lambda index: var_14
    var_16 = module_0.Token(var_12)
    var_17 = error.messages()[0]
    var_18 = 'invalid'
    var_19 = lambda _: var_7
    var_20 = module_0.Token(var_18)
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = error.messages()[0]



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = error.messages()[0]
    var_6 = 'valid'
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'thirty'
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'nested'
    var_15 = 'address'
    var_16 = '123 Main St'
    var_17 = {var_15: var_16}
    var_18 = {var_0: var_2, var_14: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = {}
    var_21 = {var_0: var_2, var_14: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = error.messages()[0]



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '{"name": "Alice", "age": 25}'
    var_1 = '{"name": "ThisNameIsTooLong", "age": 25}'
    var_2 = error.messages()[0]
    var_3 = '{"name": "Alice"}'
    var_4 = error.messages()[0]
    var_5 = '{"name": "ThisNameIsTooLong", "age": 15}'
    var_6 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = error.messages()[0]
    var_8 = 25
    var_9 = {var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = error.messages()[0]



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John Doe'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'John Doe John Doe'
    var_10 = {var_0: var_9, var_1: var_3}
    var_11 = module_0.Token(var_10)
    var_12 = -5
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = module_0.Token(var_13)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_1: var_3}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = error.messages()[0]
    var_10 = ''
    var_11 = -1
    var_12 = {var_0: var_10, var_8: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = 'All tests passed.'
    var_16 = print(var_15)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 0
    var_7 = {var_5: var_6}
    var_8 = 20
    var_9 = {var_5: var_8}
    var_10 = module_0.Token(var_4)
    var_11 = {var_0: var_2}
    var_12 = {var_5: var_6}
    var_13 = 15
    var_14 = {var_5: var_13}
    var_15 = module_0.Token(var_11)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = '30'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_5)
    var_7 = lambda index: var_6
    var_8 = module_0.Token(var_4)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'unexpected'
    var_8 = module_0.Token(var_7)
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = 123
    var_12 = 'abc'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = 'nested'
    var_16 = {var_0: var_11}
    var_17 = {var_15: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = 'All test cases passed!'
    var_20 = print(var_19)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'invalid'
    var_5 = 7
    var_6 = module_0.Token(var_4)
    var_7 = error.messages()[0]
    var_8 = ''
    var_9 = module_0.Token(var_8)
    var_10 = error.messages()[0]
    var_11 = 'All test cases passed!'
    var_12 = print(var_11)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)



