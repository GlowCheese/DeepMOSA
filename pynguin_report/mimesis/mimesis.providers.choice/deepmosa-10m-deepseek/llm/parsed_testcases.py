####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_with_list_and_length_one. Retrieved 8/9 statements.
# Partially parsed test_choice_with_string_and_length. Retrieved 5/8 statements.
# Partially parsed test_choice_with_tuple_and_length. Retrieved 8/11 statements.
# Partially parsed test_choice_with_unique_elements. Retrieved 8/11 statements.


import mimesis.providers.choice as module_0


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
    var_6 = bool(var_5 in var_4)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0]
    var_9 = bool(var_6[0] in var_4)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abc'
    var_2 = 2
    var_3 = var_0(items=var_1, length=var_2)
    var_4 = len(var_3)
    assert var_4 == 2


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 5
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 5


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'aabbbccccddddd'
    var_2 = 4
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = set(var_4)
    var_7 = len(var_6)
    assert var_7 == 4


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 123
    var_3 = var_0(items=var_2)
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = []
    var_3 = var_0(items=var_2)
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = -1
    var_7 = var_0(items=var_5, length=var_6)
    var_8 = True
    var_9 = bool(var_8)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 'aa'
    var_3 = 3
    var_4 = True
    var_5 = var_0(items=var_2, length=var_3, unique=var_4)
    var_6 = True
    var_7 = bool(var_6)
    assert var_7 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_with_list_and_length_one. Retrieved 8/9 statements.
# Partially parsed test_choice_with_string_and_length. Retrieved 5/8 statements.
# Partially parsed test_choice_with_tuple_and_length. Retrieved 8/11 statements.
# Partially parsed test_choice_with_unique_elements. Retrieved 8/11 statements.



def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
    var_6 = bool(var_5 in var_4)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0]
    var_9 = bool(var_6[0] in var_4)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abc'
    var_2 = 2
    var_3 = var_0(items=var_1, length=var_2)
    var_4 = len(var_3)
    assert var_4 == 2


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 5
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 5


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'aabbbccccddddd'
    var_2 = 4
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = set(var_4)
    var_7 = len(var_6)
    assert var_7 == 4


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 123
    var_3 = var_0(items=var_2)
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = []
    var_3 = var_0(items=var_2)
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = -1
    var_7 = var_0(items=var_5, length=var_6)
    var_8 = True
    var_9 = bool(var_8)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Choice()
    var_1 = False
    var_2 = 'aa'
    var_3 = 3
    var_4 = True
    var_5 = var_0(items=var_2, length=var_3, unique=var_4)
    var_6 = True
    var_7 = bool(var_6)
    assert var_7 is True



