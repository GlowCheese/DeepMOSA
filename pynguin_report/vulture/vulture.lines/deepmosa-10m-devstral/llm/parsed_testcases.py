####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.lines as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = ()
    var_2 = 'decorator_list'
    var_3 = ()
    var_4 = 'lineno'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = type(var_0, var_3, var_6)
    var_8 = var_7()
    var_9 = [var_8]
    var_10 = {var_2: var_9}
    var_11 = type(var_0, var_1, var_10)
    var_12 = var_11()
    var_13 = module_0.get_first_line_number(var_12)
    assert var_13 == 10

import vulture.lines as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = ()
    var_2 = 'lineno'
    var_3 = 5
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = module_0.get_first_line_number(var_6)
    assert var_7 == 5



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_first_line_number_with_decorators. Retrieved 3/15 statements.
# Partially parsed test_get_first_line_number_without_decorators. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 7

def test_case_0():
    var_0 = 10



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_first_line_number_with_decorators. Retrieved 3/15 statements.
# Partially parsed test_get_first_line_number_without_decorators. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 10

def test_case_0():
    var_0 = 10



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_first_line_number_with_decorators. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 10



# Parsed testcases at query #4
#--------------------------




import vulture.lines as module_0

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'decorator_list'
    var_3 = 'Decorator'
    var_4 = ()
    var_5 = 'lineno'
    var_6 = 5
    var_7 = {var_5: var_6}
    var_8 = type(var_3, var_4, var_7)
    var_9 = var_8()
    var_10 = [var_9]
    var_11 = {var_2: var_10}
    var_12 = type(var_0, var_1, var_11)
    var_13 = module_0.get_first_line_number(var_12)
    assert var_13 == 5

import vulture.lines as module_0

def test_case_0():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'lineno'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = module_0.get_first_line_number(var_5)
    assert var_6 == 10



