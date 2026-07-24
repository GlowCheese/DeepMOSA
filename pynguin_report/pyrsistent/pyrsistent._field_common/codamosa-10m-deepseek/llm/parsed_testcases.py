####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = {}
    var_6 = '_fields'
    var_7 = 'field6'
    var_8 = 6
    var_9 = {var_7: var_8}
    var_10 = 'field1'
    var_11 = 'field2'
    var_12 = 10
    var_13 = 20
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 100
    var_16 = 200
    var_17 = {}
    var_18 = 1000
    var_19 = 2000
    var_20 = {var_10: var_18, var_11: var_19}



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = {var_2: var_0}
    var_6 = {var_2: var_0}
    var_7 = {var_2: var_0}



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = {}
    var_3 = {}
    var_4 = {}
    var_5 = 'one'
    var_6 = {var_1: var_5}
    var_7 = {var_1: var_5}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'value'
    var_2 = 'fmt'
    var_3 = 'val'



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '__fields__'
    var_6 = 'c'
    var_7 = None
    var_8 = module_0._PField(var_7, var_7, var_7, var_7, var_7, var_7)
    var_9 = {var_0: var_2, var_1: var_3, var_6: var_8}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'data'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a is not 1'
    var_6 = lambda x: (x[var_0] == var_2, var_5)
    var_7 = 'b is not 2'
    var_8 = lambda x: (x[var_1] == var_3, var_7)
    var_9 = [var_6, var_8]
    var_10 = module_0.check_global_invariants(var_4, var_9)
    var_11 = 'a'
    var_12 = 2
    var_13 = 'a is not 2'
    var_14 = lambda x: (x[var_11] == var_12, var_13)
    var_15 = [var_14]
    var_16 = module_0.check_global_invariants(var_4, var_15)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 4
    var_3 = 'Should be 4'
    var_4 = lambda x: (x(var_0) == var_2, var_3)
    var_5 = 3
    var_6 = 6
    var_7 = 'Should be 6'
    var_8 = lambda x: (x(var_5) == var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_1, var_9)
    var_11 = 2
    var_12 = 4
    var_13 = 'Should be 4'
    var_14 = lambda x: (x(var_11) == var_12, var_13)
    var_15 = 3
    var_16 = 5
    var_17 = 'Should be 5'
    var_18 = lambda x: (x(var_15) == var_16, var_17)
    var_19 = [var_14, var_18]
    var_20 = module_0.check_global_invariants(var_1, var_19)
    var_21 = 2
    var_22 = 5
    var_23 = 'Should be 5'
    var_24 = lambda x: (x(var_21) == var_22, var_23)
    var_25 = 3
    var_26 = 6
    var_27 = 'Should be 6'
    var_28 = lambda x: (x(var_25) == var_26, var_27)
    var_29 = [var_24, var_28]
    var_30 = module_0.check_global_invariants(var_1, var_29)
    var_31 = 2
    var_32 = 4
    var_33 = 'Should be 4'
    var_34 = lambda x: (x(var_31) == var_32, var_33)
    var_35 = 3
    var_36 = 5
    var_37 = 'Should be 5'
    var_38 = lambda x: (x(var_35) == var_36, var_37)
    var_39 = [var_34, var_38]
    var_40 = module_0.check_global_invariants(var_1, var_39)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda _: var_2
    var_4 = False
    var_5 = lambda x: x
    var_6 = lambda _, value: value
    var_7 = (var_0, var_1)
    var_8 = lambda _: var_7
    var_9 = lambda x, ignore_extra=False: x
    var_10 = lambda _, value: value
    var_11 = (var_0, var_1)
    var_12 = lambda _: var_11
    var_13 = lambda x, ignore_extra=False: x
    var_14 = lambda _, value: value



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'base1_field1'
    var_1 = 'base1_field2'
    var_2 = 'base2_field2'
    var_3 = 'base2_field3'
    var_4 = 'field4'
    var_5 = 'test_field4'
    var_6 = {var_4: var_5}
    var_7 = '__fields__'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}



# Parsed testcases at query #15
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)
    var_6 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'test_value'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'json'
    var_2 = lambda _, value: value
    var_3 = module_0.serialize(var_2, var_1, var_0)
    var_4 = None
    var_5 = lambda _, value: var_4
    var_6 = module_0.serialize(var_5, var_1, var_0)
    assert var_6 is None



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = True



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '_fields'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {}
    var_5 = '_fields'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = lambda x, ignore_extra=None: x
    var_6 = lambda x: x
    var_7 = (var_0, var_1)
    var_8 = lambda x: var_7
    var_9 = lambda x: x
    var_10 = lambda x: x



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = []



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = lambda f, v: f + v
    var_1 = 'format'
    var_2 = 'value'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'a'
    var_4 = {var_3: var_0}



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 0



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = {}
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_4: var_0}
    var_7 = {}
    var_8 = {}
    var_9 = {var_4: var_0}



# Parsed testcases at query #27
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #29
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = module_0.serialize(var_0)
    var_2 = 'value'



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None
    var_7 = {}
    var_8 = module_0.pmap(var_7)
    var_9 = 'a'
    var_10 = 'not an int'
    var_11 = {var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.pmap(var_13)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'All tests passed for check_global_invariants'
    var_1 = print(var_0)



# Parsed testcases at query #33
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ' custom '
    var_1 = lambda format, value: format + var_0 + str(value)
    var_2 = 'format'
    var_3 = 123
    var_4 = module_0.serialize(var_1, var_2, var_3)
    assert var_4 == 'format custom 123'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 'test_field'
    var_3 = 'test_field'
    var_4 = 'any value'
    var_5 = 'A'
    var_6 = {var_5}
    var_7 = 'test_field'
    var_8 = 'test_field'
    var_9 = 'test_field'



# Parsed testcases at query #35
#--------------------------




# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Base1_field1'
    var_1 = 'Base1_field2'
    var_2 = 'Base2_field2'
    var_3 = 'Base2_field3'
    var_4 = 'field4'
    var_5 = 'TestClass_field4'
    var_6 = {var_4: var_5}
    var_7 = '__annotations__'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'base1_field1'
    var_1 = 'base1_field2'
    var_2 = 'base2_field2'
    var_3 = 'base2_field3'
    var_4 = 'field4'
    var_5 = 'test_field4'
    var_6 = {var_4: var_5}
    var_7 = '_precord_fields'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'OK'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = 0
    var_8 = 'Value must be positive'
    var_9 = lambda x: (x.value > var_7, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = 5
    var_12 = False
    var_13 = 'Error 1'
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = False
    var_17 = 'Error 2'
    var_18 = (var_16, var_17)
    var_19 = lambda x: var_18
    var_20 = [var_15, var_19]
    var_21 = (var_0, var_1)
    var_22 = lambda x: var_21
    var_23 = False
    var_24 = 'Only error'
    var_25 = (var_23, var_24)
    var_26 = lambda x: var_25
    var_27 = (var_0, var_4)
    var_28 = lambda x: var_27
    var_29 = [var_22, var_26, var_28]



# Parsed testcases at query #40
#--------------------------




# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = 'All test cases passed'
    var_2 = print(var_1)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = ''
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = None
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = {var_6: var_1}
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_10}



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = set()
    var_1 = lambda x: x
    var_2 = False
    var_3 = set()
    var_4 = lambda x: x
    var_5 = True
    var_6 = set()
    var_7 = lambda x: x
    var_8 = set()



# Parsed testcases at query #44
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 'Value must be greater than 0'
    var_3 = (var_1, var_2)
    var_4 = 10
    var_5 = lambda x: x < var_4
    var_6 = 'Value must be less than 10'
    var_7 = (var_5, var_6)
    var_8 = [var_3, var_7]
    var_9 = 5
    var_10 = module_0.check_global_invariants(var_9, var_8)
    var_11 = -1
    var_12 = module_0.check_global_invariants(var_11, var_8)
    var_13 = 15
    var_14 = module_0.check_global_invariants(var_13, var_8)
    var_15 = -5
    var_16 = module_0.check_global_invariants(var_15, var_8)



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = True
    var_4 = lambda x: x
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'base1_field1'
    var_1 = 'base1_field2'
    var_2 = 'base2_field2'
    var_3 = 'base2_field3'
    var_4 = 'field4'
    var_5 = 'test_field4'
    var_6 = {var_4: var_5}
    var_7 = '__fields__'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Base1Field1'
    var_1 = 'Base2Field2'
    var_2 = 'field3'
    var_3 = 'TestClassField3'
    var_4 = {var_2: var_3}
    var_5 = '_fields'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Test the check_global_invariants function.\n    '
    var_1 = 1
    var_2 = []
    var_3 = 1
    var_4 = 1



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {}
    var_3 = '_fields'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_field'
    var_2 = 'test_field'
    var_3 = int()
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_3: var_0, var_4: var_5, var_7: var_8}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'field6'
    var_1 = '_fields'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = '_fields'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {}
    var_3 = '__fields__'
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = 3
    var_8 = {}
    var_9 = {var_4: var_5}



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_field'
    var_2 = 'invalid_type'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = None
    var_3 = {}
    var_4 = {}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_1, var_6: var_7}
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_5: var_1, var_6: var_7, var_9: var_10}
    var_12 = 'not a map'
    var_13 = 1
    var_14 = {var_13: var_13}
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_16}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'test_value'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda format, value: str(value)
    var_2 = 'json'
    var_3 = module_0.serialize(var_1, var_2, var_0)
    var_4 = str(var_0)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = 'bad'
    var_6 = 1
    var_7 = {var_5: var_6}



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_field'
    var_2 = 'invalid_type'
    var_3 = 'Test failed: Expected PTypeError'
    var_4 = print(var_3)
    var_5 = 'All tests passed'
    var_6 = print(var_5)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda _: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda _: var_4
    var_6 = [var_3, var_5]
    var_7 = False
    var_8 = 'error1'
    var_9 = (var_7, var_8)
    var_10 = lambda _: var_9
    var_11 = 'error2'
    var_12 = (var_7, var_11)
    var_13 = lambda _: var_12
    var_14 = [var_10, var_13]
    var_15 = module_0.check_global_invariants(var_0, var_14)
    var_16 = (var_0, var_15)
    var_17 = lambda _: var_16
    var_18 = (var_7, var_8)
    var_19 = lambda _: var_18
    var_20 = [var_17, var_19]
    var_21 = module_0.check_global_invariants(var_0, var_20)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = module_0.field(initial=var_1)
    var_3 = True
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 'not an int'
    var_6 = 'not callable'
    var_7 = module_0.field(invariant=var_6)
    var_8 = 'not callable'
    var_9 = module_0.field(factory=var_8)
    var_10 = 'not callable'
    var_11 = module_0.field(serializer=var_10)
    var_12 = 42
    var_13 = module_0.field(var_12)
    var_14 = 'All field tests passed'
    var_15 = print(var_14)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda format, value: value * var_0
    var_2 = 'format'
    var_3 = 5
    var_4 = module_0.serialize(var_1, var_2, var_3)
    assert var_4 == 10
    var_5 = 'hello'
    var_6 = module_0.serialize(var_1, var_2, var_5)
    assert var_6 == 'hellohello'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'regular_value'
    var_2 = 'fmt'
    var_3 = 123



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 10
    var_2 = 'test_field'
    var_3 = 'invalid'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 'value3'
    var_3 = 'field3'
    var_4 = 'value3'
    var_5 = {var_3: var_4}
    var_6 = '_fields'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'a'
    var_4 = {var_3: var_0}



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 3.14
    var_5 = 'any value'
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'regular_value'
    var_2 = 'fmt'
    var_3 = 123



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = None



# Parsed testcases at query #28
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = '_'
    var_2 = lambda f, v: f + var_1 + v
    var_3 = 'value'
    var_4 = module_0.serialize(var_2, var_0, var_3)
    assert var_4 == 'format_value'



# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = True
    var_3 = lambda x: x
    var_4 = module_0.field(factory=var_3)
    var_5 = lambda x, ignore_extra=False: x
    var_6 = module_0.field(factory=var_5)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 'test_field'
    var_3 = 'test_field'
    var_4 = 'string'
    var_5 = 'string'
    var_6 = 123



# Parsed testcases at query #32
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)
    assert var_2 is None
    var_3 = 4
    var_4 = module_0.check_global_invariants(var_3, var_1)
    var_5 = -1
    var_6 = module_0.check_global_invariants(var_5, var_1)



# Parsed testcases at query #33
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #34
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, value: value
    var_1 = 'format'
    var_2 = 'value'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'value'



# Parsed testcases at query #35
#--------------------------




# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = None
    var_6 = {}
    var_7 = {var_1: var_4}



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 'test_field'
    var_3 = None
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_3: var_0, var_4: var_5, var_7: var_8}



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'field_name'
    var_2 = 'field_name'
    var_3 = 'not_a_record'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 10
    var_2 = 'hello'
    var_3 = 'test_field'
    var_4 = 10.5
    var_5 = 10.5



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'test_format'
    var_1 = 'custom_format'
    var_2 = 'test_value'
    var_3 = 'test_format'
    var_4 = 'test_value'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'base1_field1'
    var_1 = 'base1_field2'
    var_2 = 'base2_field2'
    var_3 = 'base2_field3'
    var_4 = 'test_field3'
    var_5 = 'test_field4'
    var_6 = '_fields'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 15
    var_3 = -15



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'valid_string'
    var_2 = 'test_field'
    var_3 = 'invalid_string'



# Parsed testcases at query #45
#--------------------------




# Parsed testcases at query #46
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_format'
    var_1 = lambda f, v: f + v
    var_2 = 'test_value'
    var_3 = module_0.serialize(var_1, var_0, var_2)
    assert var_3 == 'test_formattest_value'
    var_4 = 'test_serialize passed'
    var_5 = print(var_4)



# Parsed testcases at query #47
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'format'
    var_1 = 'value'



# Parsed testcases at query #49
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_6]
    var_8 = module_0.check_global_invariants(var_2, var_7)
    assert var_8 is None
    var_9 = {var_0: var_1}
    var_10 = False
    var_11 = 'error_code'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_13]
    var_15 = module_0.check_global_invariants(var_9, var_14)
    var_16 = {var_15: var_1}
    var_17 = (var_3, var_4)
    var_18 = lambda x: var_17
    var_19 = (var_10, var_11)
    var_20 = lambda x: var_19
    var_21 = [var_18, var_20]
    var_22 = module_0.check_global_invariants(var_16, var_21)
    var_23 = {var_22: var_1}
    var_24 = 'error_code1'
    var_25 = (var_10, var_24)
    var_26 = lambda x: var_25
    var_27 = 'error_code2'
    var_28 = (var_10, var_27)
    var_29 = lambda x: var_28
    var_30 = [var_26, var_29]
    var_31 = module_0.check_global_invariants(var_23, var_30)
    var_32 = 'All test cases passed'
    var_33 = print(var_32)



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 15
    var_3 = -5



