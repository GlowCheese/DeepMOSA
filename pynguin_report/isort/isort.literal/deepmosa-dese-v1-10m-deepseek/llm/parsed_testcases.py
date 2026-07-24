####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]
    var_7 = module_1._list(var_6, var_2)
    assert var_7 == '[1, 2, 3]'
    var_8 = 'c'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = [var_8, var_9, var_10]
    var_12 = module_1._list(var_11, var_2)
    assert var_12 == "['a', 'b', 'c']"
    var_13 = []
    var_14 = module_1._list(var_13, var_2)
    assert var_14 == '[]'



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = set()
    var_4 = module_1._set(var_3, var_2)
    assert var_4 == '{}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 1
    var_4 = {var_3}
    var_5 = module_1._set(var_4, var_2)
    assert var_5 == '{1}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 2
    var_5 = 1
    var_6 = {var_3, var_4, var_5}
    var_7 = module_1._set(var_6, var_2)
    assert var_7 == '{1, 2, 3}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'c'
    var_4 = 'b'
    var_5 = 'a'
    var_6 = {var_3, var_4, var_5}
    var_7 = module_1._set(var_6, var_2)
    assert var_7 == "{'a', 'b', 'c'}"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = True
    var_6 = {var_3, var_4, var_5}
    var_7 = module_1._set(var_6, var_2)
    assert var_7 == "{1, True, 'a'}"



# Parsed testcases at query #3
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\nb = 2\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1b = 2c = 3'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\n\nb = 2\n\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1b = 2c = 3'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1 \nb = 2 \nc = 3 '
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1 b = 2 c = 3 '

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\nb\nc = 3'
    var_1 = module_0.assignments(var_0)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1'

import isort.literal as module_0

def test_case_0():
    var_0 = 'c = 3\na = 1\nb = 2'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1b = 2c = 3'



# Parsed testcases at query #4
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\nb = 2\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'

import isort.literal as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == ''

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'invalid_sort_type'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [3, 1, 2]'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2, 3]   \n'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a = [3, 1, 2]'
    var_1 = 80
    var_2 = lambda x, _, __: x.upper()
    var_3 = module_0.Config()
    var_4 = 'list'
    var_5 = 'py'
    var_6 = module_1.assignment(var_0, var_4, var_5, var_3)
    assert var_6 == 'A = [1, 2, 3]'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'b'
    var_4 = 'a'
    var_5 = 2
    var_6 = 1
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = "{'a': 1, 'b': 2}"
    var_9 = module_1._dict(var_7, var_2)



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = ()
    var_4 = module_1._tuple(var_3, var_2)
    assert var_4 == '()'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 1
    var_4 = (var_3,)
    var_5 = module_1._tuple(var_4, var_2)
    assert var_5 == '(1,)'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = (var_3, var_4, var_5)
    var_7 = module_1._tuple(var_6, var_2)
    assert var_7 == '(1, 2, 3)'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'b'
    var_4 = 'a'
    var_5 = 'c'
    var_6 = (var_3, var_4, var_5)
    var_7 = module_1._tuple(var_6, var_2)
    assert var_7 == "('a', 'b', 'c')"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 'a'
    var_5 = 1
    var_6 = 'b'
    var_7 = (var_3, var_4, var_5, var_6)
    var_8 = module_1._tuple(var_7, var_2)
    assert var_8 == "(1, 3, 'a', 'b')"



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = lambda x, y, z: x.upper()
    var_2 = module_0.Config()
    var_3 = 'var = [3, 1, 2]'
    var_4 = 'lists'
    var_5 = '.py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5, var_5, var_3]
    var_7 = module_1._unique_list(var_6, var_2)
    assert var_7 == '[1, 2, 3]'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_assignment_with_invalid_literal_raises_literal_parsing_failure. Retrieved 4/6 statements.


import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = (var_3, var_4, var_5, var_4, var_5)
    var_7 = module_1._unique_tuple(var_6, var_2)
    assert var_7 == '(1, 2, 3)'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = "x = 'invalid_literal'"
    var_3 = 'invalid_type'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 'x = invalid_literal'
    var_3 = 'int'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)



# Parsed testcases at query #13
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = 1'

import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\n\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid line'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'some_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "x = 'string'"
    var_1 = 'some_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = 1'
    var_3 = 'numbers'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = 1'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'x = 1'
    var_3 = 'numbers'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = 1'



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = lambda code, ext, cfg: code.upper()
    var_2 = module_0.Config()
    var_3 = 'var = [3, 1, 2]'
    var_4 = 'list'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = lambda code, ext, cfg: code.upper()
    var_2 = module_0.Config()
    var_3 = 'x = [2, 1, 3]'
    var_4 = 'assignments'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)
    assert var_6 == 'X = [2, 1, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = None
    var_2 = module_0.Config()
    var_3 = 'x = [2, 1, 3]'
    var_4 = 'assignments'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)
    assert var_6 == 'x = [1, 2, 3]'



# Parsed testcases at query #17
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid_literal = this_will_fail'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, extension, config: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'lists'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)



# Parsed testcases at query #19
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [2, 1, 3]'
    var_1 = 'lists'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'lists'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #20
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'unsupported'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "a = {'c': 3, 'a': 1}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "a = {'a': 1, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = {3, 1, 2}'
    var_1 = 'set'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = {1, 2, 3}'

import isort.literal as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == ''

import isort.literal as module_0

def test_case_0():
    var_0 = '   \n   '
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == ''

import isort.literal as module_0

def test_case_0():
    var_0 = 'a 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #21
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x.upper()
    var_1 = 80
    var_2 = module_0.Config()
    var_3 = 'x = [2, 1]'
    var_4 = 'sequences'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)
    assert var_6 == 'X = [1, 2]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = None
    var_1 = 80
    var_2 = module_0.Config()
    var_3 = 'x = [2, 1]'
    var_4 = 'sequences'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)
    assert var_6 == 'x = [1, 2]'



# Parsed testcases at query #22
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'lists'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "x = {'b': 2, 'a': 1}"
    var_1 = 'dicts'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "x = {'a': 1, 'b': 2}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = {3, 1, 2}'
    var_1 = 'sets'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = {1, 2, 3}'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = (3, 1, 2)'
    var_1 = 'tuples'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = (1, 2, 3)'



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = 88
    var_4 = module_0.Config()
    var_5 = module_1.assignment(var_0, var_1, var_2, var_4)



# Parsed testcases at query #24
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'invalid_sort_type'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [3, 1, 2]'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a = [3, 1, 2]'
    var_1 = lambda x, y, z: x.upper()
    var_2 = module_0.Config()
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_0, var_3, var_4, var_2)
    assert var_5 == 'A = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [3, 1, 2]  \n'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2, 3]  \n'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\n a 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = 'invalid_sort'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = 'dict'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'b'
    var_1 = 'a'
    var_2 = 'c'
    var_3 = 2
    var_4 = 1
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = "{'a': 1, 'b': 2, 'c': 3}"
    var_8 = 88
    var_9 = module_0.Config()
    var_10 = module_1.ISortPrettyPrinter(var_9)
    var_11 = module_1._dict(var_6, var_10)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = {}
    var_1 = '{}'
    var_2 = 88
    var_3 = module_0.Config()
    var_4 = module_1.ISortPrettyPrinter(var_3)
    var_5 = module_1._dict(var_0, var_4)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = "{'a': 1}"
    var_4 = 88
    var_5 = module_0.Config()
    var_6 = module_1.ISortPrettyPrinter(var_5)
    var_7 = module_1._dict(var_2, var_6)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = "{'a': 1, 'b': 2, 'c': 3}"
    var_8 = 88
    var_9 = module_0.Config()
    var_10 = module_1.ISortPrettyPrinter(var_9)
    var_11 = module_1._dict(var_6, var_10)



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = (var_3, var_4, var_5, var_5, var_4)
    var_7 = module_1._unique_tuple(var_6, var_2)
    assert var_7 == '(1, 2, 3)'



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5, var_3, var_4]
    var_7 = module_1._unique_list(var_6, var_2)
    assert var_7 == '[1, 2, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]
    var_7 = module_1._unique_list(var_6, var_2)
    assert var_7 == '[1, 2, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = []
    var_4 = module_1._unique_list(var_3, var_2)
    assert var_4 == '[]'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = set()
    var_4 = module_1._set(var_3, var_2)
    assert var_4 == '{}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = {var_3, var_4, var_5}
    var_7 = module_1._set(var_6, var_2)
    assert var_7 == '{1, 2, 3}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = {var_3, var_4, var_5}
    var_7 = module_1._set(var_6, var_2)
    assert var_7 == '{1, 2, 3}'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_assignment_with_formatting_function. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'x = 1'
    var_2 = 'assignments'
    var_3 = 'py'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_assignment_raises_literal_parsing_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__tuple_sorts_and_formats_tuple. Retrieved 4/8 statements.
# Partially parsed test__tuple_empty_tuple. Retrieved 2/6 statements.
# Partially parsed test__tuple_single_element. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = tuple()
    var_1 = tuple()

def test_case_0():
    var_0 = 42
    var_1 = (var_0,)



# Parsed testcases at query #9
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'invalid_sort_type'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [1, 2, 3]'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = (var_3, var_4, var_5)
    var_7 = module_1._tuple(var_6, var_2)
    assert var_7 == '(1, 2, 3)'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'dict'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = module_0.Config()
    var_3 = 'x = invalid_literal'
    var_4 = 'strings'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)



# Parsed testcases at query #13
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'a = 1\nb = 2'
    var_2 = 'assignments'
    var_3 = '.py'
    var_4 = module_0.assignment(var_0, var_2, var_3)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'int'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "a = 'string'"
    var_1 = 'int'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "a = {'b': 2, 'a': 1}"
    var_1 = "a = {'a': 1, 'b': 2}"
    var_2 = 'dict'
    var_3 = '.py'
    var_4 = module_0.assignment(var_0, var_2, var_3)

import isort.literal as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'int'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_formatting_function_applied_when_present. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'assignments'
    var_2 = '.py'



# Parsed testcases at query #15
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1b = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'invalid_sort_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2]'

import isort.literal as module_0

def test_case_0():
    var_0 = "a = {'b': 2, 'a': 1}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "a = {'a': 1, 'b': 2}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]  \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = [1, 2]  \n'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = lambda x, y, z: x.upper()
    var_2 = module_0.Config()
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_0, var_3, var_4, var_2)
    assert var_5 == 'A = [1, 2]'



# Parsed testcases at query #16
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\n'
    var_1 = 'invalid_sort_type'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'a = 1\n'
    var_1 = lambda code, ext, cfg: code.upper()
    var_2 = module_0.Config()
    var_3 = 'assignments'
    var_4 = 'py'
    var_5 = module_1.assignment(var_0, var_3, var_4, var_2)
    assert var_5 == 'A = 1\n'



# Parsed testcases at query #17
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid_literal = not_a_valid_literal'
    var_1 = 'int'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = 1'
    var_3 = 'numbers'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'x = 1'
    var_3 = 'numbers'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)



# Parsed testcases at query #19
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'a = 1\nb = 2'
    var_2 = 'assignments'
    var_3 = '.py'
    var_4 = module_0.assignment(var_0, var_2, var_3)

import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid line'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = 1'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = "a = {'b': 2, 'a': 1}"
    var_1 = lambda x, _, __: x.upper()
    var_2 = module_0.Config()
    var_3 = "A = {'A': 1, 'B': 2}"
    var_4 = 'dict'
    var_5 = '.py'
    var_6 = module_1.assignment(var_0, var_4, var_5, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "a = {'b': 2, 'a': 1}  \n"
    var_1 = "a = {'a': 1, 'b': 2}  \n"
    var_2 = 'dict'
    var_3 = '.py'
    var_4 = module_0.assignment(var_0, var_2, var_3)



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'x = 42'
    var_1 = 'int'
    var_2 = 'py'
    var_3 = module_0.Config()
    var_4 = module_1.assignment(var_0, var_1, var_2, var_3)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_formatting_function_evaluates_to_true. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'var = [2, 1]'
    var_2 = 'lists'
    var_3 = '.py'
    var_4 = module_1.assignment(var_1, var_2, var_3, var_0)
    assert var_4 == 'var = [1, 2]'



# Parsed testcases at query #22
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'

import isort.literal as module_0

def test_case_0():
    var_0 = 'a + 2'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = [2, 1]'
    var_1 = 'invalid_sort_type'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'a = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "a = {'key': 'value'}"
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, extension, config: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'lists'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'



