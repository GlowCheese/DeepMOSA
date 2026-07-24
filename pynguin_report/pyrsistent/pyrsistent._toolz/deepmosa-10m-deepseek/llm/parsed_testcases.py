####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._toolz as module_0


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]
    var_15 = 0
    var_16 = [var_0, var_15, var_0]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 6


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_4]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 'Alice'


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 is None


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'apple'
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 is None


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 is None


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_14, var_17)
    assert var_18 == 0


def test_case_0():
    var_0 = {}
    var_1 = 'y'
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.get_in(var_2, var_0, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = None
    var_5 = True
    var_6 = module_0.get_in(var_3, var_2, var_4, var_5)
    assert var_6 == 1



# Parsed testcases at query #3
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.get_in(var_1, var_3, no_default=var_4)
    assert var_5 == 1



# Parsed testcases at query #4
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = None
    var_5 = True
    var_6 = module_0.get_in(var_3, var_2, var_4, var_5)
    assert var_6 == 1



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = 0
    var_13 = [var_12, var_0, var_12]
    var_14 = module_0.get_in(var_13, var_11)
    assert var_14 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_4]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 10


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 'missing'
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 'missing'


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'a'
    var_6 = 'c'
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = module_0.get_in(var_7, var_4, no_default=var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2


def test_case_0():
    var_0 = 'nums'
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 2
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 30



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_in_with_freeze_example. Retrieved 30/39 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = 0
    var_13 = [var_12, var_0, var_12]
    var_14 = module_0.get_in(var_13, var_11)
    assert var_14 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = [var_0, var_6, var_1]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 5


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = [var_0]
    var_18 = 'total'
    var_19 = [var_1, var_18]
    var_20 = 'apple'
    var_21 = [var_1, var_4, var_20]
    var_22 = 10
    var_23 = [var_1, var_4, var_22]
    var_24 = [var_1, var_18]
    var_25 = 'y'
    var_26 = [var_25]
    var_27 = {}
    var_28 = True
    var_29 = module_0.get_in(var_26, var_27, no_default=var_28)
    var_30 = bool(False)
    assert var_30 is True
    var_31 = bool(True)
    assert var_31 is True



# Parsed testcases at query #7
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = None
    var_7 = True
    var_8 = module_0.get_in(var_5, var_4, var_6, var_7)
    assert var_8 == 1



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = 0
    var_13 = [var_12, var_0, var_12]
    var_14 = module_0.get_in(var_13, var_11)
    assert var_14 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_4]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 5
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 5


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = 0
    var_7 = module_0.get_in(var_5, var_3, var_6)
    assert var_7 == 0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 10
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 5
    var_5 = module_0.get_in(var_3, var_2, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = True
    var_5 = module_0.get_in(var_3, var_2, no_default=var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 'Alice'


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 is None


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_14, var_17)
    assert var_18 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 0
    var_8 = [var_7, var_7, var_0]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 2


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = [var_0, var_6, var_1]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 5


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 'missing'
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 'missing'


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'a'
    var_6 = 'c'
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = module_0.get_in(var_7, var_4, no_default=var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_in_with_freeze_example. Retrieved 17/19 statements.
# Partially parsed test_get_in_with_freeze_example_default. Retrieved 18/20 statements.
# Partially parsed test_get_in_with_freeze_example_no_default_raises. Retrieved 18/21 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = 0
    var_13 = [var_12, var_0, var_12]
    var_14 = module_0.get_in(var_13, var_11)
    assert var_14 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_4]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 10


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'y'
    var_16 = [var_15]
    var_17 = True
    var_18 = bool(False)
    assert var_18 is True
    var_19 = bool(True)
    assert var_19 is True



# Parsed testcases at query #3
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = None
    var_5 = False
    var_6 = module_0.get_in(var_3, var_2, var_4, var_5)
    assert var_6 == 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_in_with_freeze_example. Retrieved 25/33 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 0
    var_8 = [var_7, var_7, var_0]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 2


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = [var_0, var_6, var_1]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 5


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = module_0.get_in(var_5, var_3)
    assert var_6 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = [var_0]
    var_18 = 'total'
    var_19 = [var_1, var_18]
    var_20 = 'apple'
    var_21 = [var_1, var_4, var_20]
    var_22 = 10
    var_23 = [var_1, var_4, var_22]
    var_24 = [var_1, var_18]



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.get_in(var_1, var_3, no_default=var_4)
    assert var_5 == 1



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_9]
    var_11 = [var_6, var_10]
    var_12 = 0
    var_13 = [var_12, var_0, var_12]
    var_14 = module_0.get_in(var_13, var_11)
    assert var_14 == 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_4]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 10


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = [var_3]
    var_5 = 100
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 100


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = -1
    var_7 = module_0.get_in(var_5, var_3, var_6)
    assert var_7 == -1


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = None
    var_5 = True
    var_6 = module_0.get_in(var_3, var_2, var_4, var_5)
    assert var_6 == 1



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = None
    var_7 = True
    var_8 = module_0.get_in(var_5, var_4, var_6, var_7)
    assert var_8 == 1



