####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_5 = 'hello'
    var_6 = 'world'
    var_7 = [var_0, var_5, var_1, var_6]
    var_8 = 1
    var_9 = 2
    var_10 = 'invalid'
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda n: (n > 0, 'Not positive')
    var_13 = [var_10, var_11, var_2]
    var_14 = 1
    var_15 = -2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = True
    var_19 = True
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = [var_19, var_17, var_2]
    var_23 = lambda n: (n > 0, 'Not positive')
    var_24 = lambda n: (n < 100, 'Too large')
    var_25 = [var_23, var_24]
    var_26 = 50
    var_27 = 99
    var_28 = [var_19, var_26, var_27]
    var_29 = 1
    var_30 = 150
    var_31 = 99
    var_32 = [var_29, var_30, var_31]
    var_33 = 'builtins.int'
    var_34 = [var_19, var_32, var_2]
    var_35 = None
    var_36 = [var_19, var_5, var_35, var_32]
    var_37 = lambda n: (n > 0, 'Not positive')
    var_38 = [var_19, var_32, var_2]
    var_39 = -1
    var_40 = [var_39]
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = 'test'
    var_45 = [var_19, var_44]



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 'c'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = 'test'
    var_10 = 'value'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = 10
    var_13 = 'internal'
    var_14 = {var_12: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = 'key'
    var_17 = 1.5
    var_18 = {var_0: var_17, var_16: var_1}
    var_19 = 'inherited'
    var_20 = {var_0: var_19}
    var_21 = module_0.pmap()
    var_22 = 0



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'str'
    var_8 = []
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = 42
    var_11 = module_0.maybe_parse_user_type(var_10)
    var_12 = None
    var_13 = module_0.maybe_parse_user_type(var_12)
    var_14 = module_1.object()
    var_15 = module_0.maybe_parse_user_type(var_14)
    var_16 = 'List[int]'
    var_17 = module_0.maybe_parse_user_type(var_16)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.InvariantException()
    var_1 = str(var_0)
    assert var_1 == ', invariant_errors=[], missing_fields=[]'
    var_2 = 'error1'
    var_3 = 'error2'
    var_4 = [var_2, var_3]
    var_5 = module_0.InvariantException(var_4)
    var_6 = str(var_5)
    assert var_6 == ', invariant_errors=[error1, error2], missing_fields=[]'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = [var_7, var_8]
    var_10 = module_0.InvariantException(missing_fields=var_9)
    var_11 = str(var_10)
    assert var_11 == ', invariant_errors=[], missing_fields=[field1, field2]'
    var_12 = 'err1'
    var_13 = 'err2'
    var_14 = [var_12, var_13]
    var_15 = 'f1'
    var_16 = 'f2'
    var_17 = 'f3'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.InvariantException(var_14, var_18)
    var_20 = str(var_19)
    assert var_20 == ', invariant_errors=[err1, err2], missing_fields=[f1, f2, f3]'
    var_21 = 'static_error'
    var_22 = str(var_19)
    assert var_22 == ', invariant_errors=[dynamic_error, static_error], missing_fields=[]'
    var_23 = []
    var_24 = []
    var_25 = module_0.InvariantException(var_23, var_24)
    var_26 = str(var_25)
    assert var_26 == ', invariant_errors=[], missing_fields=[]'
    var_27 = 'single_error'
    var_28 = [var_27]
    var_29 = 'single_field'
    var_30 = [var_29]
    var_31 = module_0.InvariantException(var_28, var_30)
    var_32 = str(var_31)
    assert var_32 == ', invariant_errors=[single_error], missing_fields=[single_field]'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = 'invalid'
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda n: (n > 0, 'Not positive')
    var_13 = [var_10, var_11, var_2]
    var_14 = 1
    var_15 = -2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = 'two'
    var_19 = [var_16, var_18, var_2]
    var_20 = [var_16, var_17, var_2]
    var_21 = True
    var_22 = True
    var_23 = {var_16: var_21, var_17: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = [var_22, var_17, var_17, var_2, var_22]
    var_26 = None
    var_27 = [var_22, var_26, var_2]
    var_28 = 'int'
    var_29 = [var_22, var_17, var_2]
    var_30 = [var_22, var_17, var_2]
    var_31 = 'invalid'
    var_32 = [var_31]
    var_33 = lambda n: (n > 0, 'Not positive')
    var_34 = [var_22, var_17, var_2]
    var_35 = -1
    var_36 = [var_35]



# Parsed testcases at query #7
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'collections.OrderedDict'
    var_3 = module_0.get_type(var_2)
    var_4 = 'dummy_module'
    var_5 = 'SomeClass'
    var_6 = ()
    var_7 = {}
    var_8 = 'dummy_module.SomeClass'
    var_9 = module_0.get_type(var_8)
    var_10 = 'os.path'
    var_11 = module_0.get_type(var_10)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = 'c'
    var_8 = 'd'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'invalid'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 1
    var_15 = 123
    var_16 = {var_14: var_15}
    var_17 = lambda k, v: (v > 0, 'Value must be positive')
    var_18 = 1
    var_19 = -1
    var_20 = {var_18: var_19}
    var_21 = lambda k, v: (v > 0, 'Positive')
    var_22 = lambda k, v: (v < 10, 'Less than 10')
    var_23 = 5
    var_24 = {var_20: var_23}
    var_25 = 1
    var_26 = 15
    var_27 = {var_25: var_26}
    var_28 = 'test'
    var_29 = {var_27: var_28}
    var_30 = 'invalid'
    var_31 = 'test'
    var_32 = {var_30: var_31}
    var_33 = {var_32: var_28}
    var_34 = None
    var_35 = {var_32: var_34}
    var_36 = {var_32: var_2}
    var_37 = {var_32: var_2}
    var_38 = 'int'
    var_39 = 'str'
    var_40 = {var_32: var_28}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 11
    var_4 = 1
    var_5 = -1
    var_6 = -1
    var_7 = 25
    var_8 = 10
    var_9 = 15
    var_10 = -5



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = 10
    var_5 = 6



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 'error1'
    var_2 = 'error2'
    var_3 = 'error3'
    var_4 = (var_1, var_2, var_3)
    var_5 = 5
    var_6 = 3
    var_7 = 2
    var_8 = 4
    var_9 = 15
    var_10 = 10



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 15
    var_3 = -5
    var_4 = 50
    var_5 = 150
    var_6 = -3



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 11
    var_4 = -2
    var_5 = 1
    var_6 = True
    var_7 = False



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 1
    var_9 = 'one'
    var_10 = {var_8: var_9}
    var_11 = 'a'
    var_12 = 'not_an_int'
    var_13 = {var_11: var_12}
    var_14 = lambda k, v: (v > 0, 'Value must be positive')
    var_15 = 5
    var_16 = 10
    var_17 = {var_13: var_15, var_1: var_16}
    var_18 = 1
    var_19 = -5
    var_20 = {var_18: var_19}
    var_21 = {var_20: var_5}
    var_22 = lambda k, v: (v > 0, 'Positive')
    var_23 = lambda k, v: (v < 100, 'Less than 100')
    var_24 = 50
    var_25 = {var_20: var_24}
    var_26 = 'x'
    var_27 = {var_26: var_20}
    var_28 = 1
    var_29 = {var_28: var_28}
    var_30 = None
    var_31 = {var_5: var_20, var_6: var_30}
    var_32 = {var_20: var_5, var_1: var_6}
    var_33 = module_0.pmap(var_32)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 15
    var_4 = -3
    var_5 = 10
    var_6 = 7
    var_7 = 3
    var_8 = 1
    var_9 = 42
    var_10 = 50
    var_11 = 150
    var_12 = 999



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 6
    var_3 = 11
    var_4 = -2
    var_5 = 10
    var_6 = 7
    var_7 = 3
    var_8 = 1
    var_9 = 0
    var_10 = 99
    var_11 = 101



# Parsed testcases at query #17
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'str'
    var_8 = []
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = ()
    var_11 = module_0.maybe_parse_user_type(var_10)
    var_12 = 42
    var_13 = module_0.maybe_parse_user_type(var_12)
    var_14 = None
    var_15 = module_0.maybe_parse_user_type(var_14)
    var_16 = module_1.object()
    var_17 = module_0.maybe_parse_user_type(var_16)
    var_18 = 1
    var_19 = 2



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = -5
    var_3 = 15
    var_4 = 10
    var_5 = -5
    var_6 = -5
    var_7 = -10
    var_8 = 0



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = True



# Parsed testcases at query #20
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '_invariants'
    var_1 = ()
    var_2 = '_checked_invariants'
    var_3 = 0
    var_4 = {}
    var_5 = module_0.store_invariants(var_4, var_1, var_2, var_0)
    var_6 = var_4[var_2]
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = {}
    var_9 = module_0.store_invariants(var_8, var_1, var_2, var_0)
    var_10 = var_8[var_2]
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = module_0.store_invariants(var_8, var_1, var_2, var_0)
    var_13 = var_8[var_2]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = {}
    var_16 = module_0.store_invariants(var_15, var_1, var_2, var_0)
    var_17 = var_15[var_2]
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {}
    var_20 = module_0.store_invariants(var_19, var_1, var_2, var_0)
    var_21 = var_19[var_2]
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = module_0.store_invariants(var_19, var_1, var_2, var_0)
    var_24 = var_19[var_2][var_3]
    var_25 = None
    var_26 = 1
    var_27 = 'not a callable'
    var_28 = {var_0: var_27}
    var_29 = '_checked_invariants'
    var_30 = '_invariants'
    var_31 = module_0.store_invariants(var_28, var_1, var_29, var_30)
    var_32 = {}
    var_33 = ()
    var_34 = module_0.store_invariants(var_32, var_33, var_30, var_29)
    var_35 = module_0.store_invariants(var_32, var_33, var_30, var_29)
    var_36 = var_32[var_30][var_3]



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '__serializer__'
    var_1 = None
    var_2 = 'builtins.str'
    var_3 = 'builtins.int'
    var_4 = '__wrapped__'
    var_5 = 1
    var_6 = 2
    var_7 = 'not callable'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = 1
    var_5 = 11
    var_6 = 'less than 10'
    var_7 = 'even'
    var_8 = 10
    var_9 = 20
    var_10 = -10
    var_11 = 200
    var_12 = 15
    var_13 = -5



# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 'custom'
    var_7 = []
    var_8 = module_0.maybe_parse_user_type(var_7)
    var_9 = ()
    var_10 = module_0.maybe_parse_user_type(var_9)
    var_11 = 42
    var_12 = module_0.maybe_parse_user_type(var_11)
    var_13 = None
    var_14 = module_0.maybe_parse_user_type(var_13)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = module_0.get_type(var_0)
    var_2 = 'collections.abc.Iterable'
    var_3 = module_0.get_type(var_2)
    var_4 = 'enum.Enum'
    var_5 = module_0.get_type(var_4)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'custom'
    var_8 = []
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = ()
    var_11 = module_0.maybe_parse_user_type(var_10)
    var_12 = 42
    var_13 = module_0.maybe_parse_user_type(var_12)
    var_14 = None
    var_15 = module_0.maybe_parse_user_type(var_14)
    var_16 = module_1.object()
    var_17 = module_0.maybe_parse_user_type(var_16)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 'str'
    var_7 = []
    var_8 = module_0.maybe_parse_user_type(var_7)
    var_9 = ()
    var_10 = module_0.maybe_parse_user_type(var_9)
    var_11 = 123
    var_12 = module_0.maybe_parse_user_type(var_11)
    var_13 = None
    var_14 = module_0.maybe_parse_user_type(var_13)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = module_0.maybe_parse_user_type(var_17)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'custom'
    var_8 = 123
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.maybe_parse_user_type(var_12)
    var_14 = []
    var_15 = module_0.maybe_parse_user_type(var_14)
    var_16 = ()
    var_17 = module_0.maybe_parse_user_type(var_16)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = 'valid'
    var_7 = {var_0: var_6}
    var_8 = 'invalid'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_10: var_6}
    var_12 = 1
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = lambda k, v: (v > 0, 'Value must be positive')
    var_16 = 5
    var_17 = {var_14: var_16}
    var_18 = 1
    var_19 = -5
    var_20 = {var_18: var_19}
    var_21 = 3.14
    var_22 = {var_20: var_21}
    var_23 = 'key'
    var_24 = 42
    var_25 = {var_23: var_24}
    var_26 = 'test'
    var_27 = {var_20: var_26}
    var_28 = 3
    var_29 = [var_20, var_1, var_28]
    var_30 = {var_20: var_2, var_1: var_3}
    var_31 = module_0.pmap(var_30)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_0, var_1, var_2]
    var_7 = 'double'
    var_8 = set()
    var_9 = 'hello'
    var_10 = None
    var_11 = [var_0, var_9, var_10]
    var_12 = [var_0, var_1]



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = {var_2: var_0, var_3: var_1}
    var_7 = {var_2: var_0, var_3: var_1}
    var_8 = 1.5
    var_9 = {var_0: var_8, var_2: var_1}
    var_10 = lambda k, v: (v > 0, 'Value must be positive')
    var_11 = 5
    var_12 = 10
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = 'test'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = module_0.pmap(var_15)
    var_17 = {}



# Parsed testcases at query #12
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_4,)
    var_6 = (var_3, var_5)
    var_7 = ()
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]
    var_10 = ()
    var_11 = 0
    var_12 = None
    var_13 = 'not a callable'
    var_14 = {var_1: var_13}
    var_15 = ()
    var_16 = '_invariants'
    var_17 = '__invariant__'
    var_18 = module_0.store_invariants(var_14, var_15, var_16, var_17)
    var_19 = {}
    var_20 = module_0.store_invariants(var_19, var_15, var_16, var_17)
    var_21 = var_19[var_16]
    var_22 = len(var_21)
    assert var_22 == 4
    var_23 = {}
    var_24 = ()
    var_25 = module_0.store_invariants(var_23, var_24, var_16, var_17)
    var_26 = ()
    var_27 = module_0.store_invariants(var_23, var_26, var_16, var_17)
    var_28 = var_23[var_16][var_11]
    var_29 = ()
    var_30 = module_0.store_invariants(var_23, var_29, var_16, var_17)
    var_31 = var_23[var_16][var_11]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_4,)
    var_6 = (var_3, var_5)
    var_7 = ()
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]
    var_10 = ()
    var_11 = 0
    var_12 = None
    var_13 = 'not a callable'
    var_14 = {var_1: var_13}
    var_15 = ()
    var_16 = '_invariants'
    var_17 = '__invariant__'
    var_18 = module_0.store_invariants(var_14, var_15, var_16, var_17)
    var_19 = {}
    var_20 = module_0.store_invariants(var_19, var_15, var_16, var_17)
    var_21 = var_19[var_16]
    var_22 = len(var_21)
    assert var_22 == 4
    var_23 = {}
    var_24 = ()
    var_25 = module_0.store_invariants(var_23, var_24, var_16, var_17)
    var_26 = ()
    var_27 = module_0.store_invariants(var_23, var_26, var_16, var_17)
    var_28 = var_23[var_16][var_11]
    var_29 = ()
    var_30 = module_0.store_invariants(var_23, var_29, var_16, var_17)
    var_31 = var_23[var_16][var_11]



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 11
    var_4 = 1
    var_5 = -2
    var_6 = 3
    var_7 = 2
    var_8 = 8
    var_9 = 0
    var_10 = -5



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_ok'
    var_2 = (var_0, var_1)
    var_3 = ()
    var_4 = 0
    var_5 = 'not a callable'
    var_6 = {var_1: var_5}
    var_7 = ()
    var_8 = '_invariants'
    var_9 = '_invariant'
    var_10 = module_0.store_invariants(var_6, var_7, var_8, var_9)
    var_11 = True
    var_12 = 'dct_inv'
    var_13 = (var_11, var_12)
    var_14 = lambda self: var_13
    var_15 = {var_9: var_14}
    var_16 = module_0.store_invariants(var_15, var_7, var_8, var_9)
    var_17 = var_15[var_8]
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ()
    var_20 = module_0.store_invariants(var_15, var_19, var_8, var_9)
    var_21 = var_15[var_8][var_4]
    var_22 = None
    var_23 = ()
    var_24 = module_0.store_invariants(var_15, var_23, var_8, var_9)
    var_25 = var_15[var_8][var_4]
    var_26 = {}
    var_27 = ()
    var_28 = module_0.store_invariants(var_26, var_27, var_8, var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_ok'
    var_2 = (var_0, var_1)
    var_3 = ()
    var_4 = 0
    var_5 = 'not a callable'
    var_6 = {var_1: var_5}
    var_7 = ()
    var_8 = '_invariants'
    var_9 = '_invariant'
    var_10 = module_0.store_invariants(var_6, var_7, var_8, var_9)
    var_11 = True
    var_12 = 'dct_inv'
    var_13 = (var_11, var_12)
    var_14 = lambda self: var_13
    var_15 = {var_9: var_14}
    var_16 = module_0.store_invariants(var_15, var_7, var_8, var_9)
    var_17 = var_15[var_8]
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ()
    var_20 = module_0.store_invariants(var_15, var_19, var_8, var_9)
    var_21 = var_15[var_8][var_4]
    var_22 = None
    var_23 = ()
    var_24 = module_0.store_invariants(var_15, var_23, var_8, var_9)
    var_25 = var_15[var_8][var_4]
    var_26 = {}
    var_27 = ()
    var_28 = module_0.store_invariants(var_26, var_27, var_8, var_9)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = '_invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = 'invariant3'
    var_9 = True
    var_10 = ()
    var_11 = (var_9, var_10)
    var_12 = lambda self: var_11
    var_13 = {var_8: var_12}
    var_14 = var_13[var_1]
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13[var_1]
    var_17 = {}
    var_18 = var_17[var_1]
    var_19 = [inv.__name__ for inv in var_18]
    var_20 = ()
    var_21 = module_0.store_invariants(var_17, var_20, var_1, var_2)
    var_22 = var_17[var_1][var_5]
    var_23 = None
    var_24 = ()
    var_25 = module_0.store_invariants(var_17, var_24, var_1, var_2)
    var_26 = var_17[var_1][var_5]
    var_27 = 'not callable'
    var_28 = {}
    var_29 = '_invariants'
    var_30 = 'invariant'
    var_31 = module_0.store_invariants(var_28, var_24, var_29, var_30)
    var_32 = {}
    var_33 = module_0.store_invariants(var_32, var_24, var_29, var_30)
    var_34 = var_32[var_29]
    var_35 = [inv.__name__ for inv in var_34]
    var_36 = 'invariant_a'
    var_37 = 'invariant_d'
    var_38 = {}
    var_39 = ()
    var_40 = module_0.store_invariants(var_38, var_39, var_29, var_30)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 6
    var_3 = 11
    var_4 = 1
    var_5 = -2
    var_6 = 3
    var_7 = 2
    var_8 = 0
    var_9 = -5



# Parsed testcases at query #19
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 20
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 'wrong'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = 1
    var_12 = 'wrong'
    var_13 = {var_11: var_12}
    var_14 = lambda k, v: (v > 0, 'Value must be positive')
    var_15 = {var_13: var_5, var_1: var_6}
    var_16 = 1
    var_17 = -10
    var_18 = {var_16: var_17}
    var_19 = {var_18: var_2}
    var_20 = {var_18: var_2, var_1: var_3}
    var_21 = module_0.pmap(var_20)
    var_22 = 3.14
    var_23 = None
    var_24 = {var_18: var_5, var_2: var_22, var_1: var_23}
    var_25 = 3
    var_26 = [var_18, var_1, var_25]
    var_27 = 'test'
    var_28 = {var_18: var_27}
    var_29 = 'c'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}
    var_6 = 'invalid'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 123
    var_11 = {var_9: var_10}
    var_12 = lambda k, v: (v > 0, 'Value must be positive')
    var_13 = 1
    var_14 = -1
    var_15 = {var_13: var_14}
    var_16 = lambda k, v: (v > 0, 'Positive')
    var_17 = lambda k, v: (v < 10, 'Less than 10')
    var_18 = 1
    var_19 = 15
    var_20 = {var_18: var_19}
    var_21 = lambda k, v: (k > 0, 'Key positive')
    var_22 = {var_20: var_2}
    var_23 = -1
    var_24 = 'a'
    var_25 = {var_23: var_24}
    var_26 = {var_25: var_2, var_1: var_3}
    var_27 = module_0.pmap(var_26)
    var_28 = {var_25: var_2}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 1
    var_9 = {var_8: var_8}
    var_10 = 'a'
    var_11 = 'not_an_int'
    var_12 = {var_10: var_11}
    var_13 = 1.5
    var_14 = {var_12: var_13, var_3: var_1}
    var_15 = lambda k, v: (v > 0, 'Value must be positive')
    var_16 = 5
    var_17 = 10
    var_18 = {var_12: var_16, var_1: var_17}
    var_19 = 1
    var_20 = -5
    var_21 = {var_19: var_20}
    var_22 = {var_21: var_2}
    var_23 = {var_21: var_2}
    var_24 = lambda self, _, k, v: (str(k), v.upper())
    var_25 = 'hello'
    var_26 = {var_21: var_25}
    var_27 = 'test'
    var_28 = {var_21: var_27}
    var_29 = 1
    var_30 = 'one'
    var_31 = (var_29, var_30)
    var_32 = [var_31]
    var_33 = None
    var_34 = {var_5: var_31, var_6: var_33}



# Parsed testcases at query #22
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 1
    var_9 = {var_8: var_8}
    var_10 = 'a'
    var_11 = 'not_int'
    var_12 = {var_10: var_11}
    var_13 = 1.5
    var_14 = {var_12: var_13, var_3: var_1}
    var_15 = lambda k, v: (v > 0, 'Value must be positive')
    var_16 = 5
    var_17 = 10
    var_18 = {var_12: var_16, var_1: var_17}
    var_19 = 1
    var_20 = -5
    var_21 = {var_19: var_20}
    var_22 = {var_21: var_5, var_1: var_6}
    var_23 = module_0.pmap(var_22)
    var_24 = 'test'
    var_25 = {var_21: var_24}
    var_26 = 'string_key'
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = 'hello'
    var_30 = None
    var_31 = {var_28: var_29, var_1: var_30}
    var_32 = {var_28: var_5}



