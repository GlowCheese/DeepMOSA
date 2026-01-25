####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = {}
    var_1 = 80
    var_2 = module_0.Config()
    var_3 = module_1.ISortPrettyPrinter(var_2)
    var_4 = module_1._dict(var_0, var_3)
    assert var_4 == '{}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_1._dict(var_5, var_2)
    assert var_6 == "{'a': 1}"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'b'
    var_4 = 'a'
    var_5 = 'c'
    var_6 = 2
    var_7 = 1
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = module_1._dict(var_9, var_2)
    assert var_10 == "{'a': 1, 'b': 2, 'c': 3}"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 'a'
    var_9 = 1
    var_10 = {var_8: var_9}
    var_11 = {var_3: var_7, var_4: var_10}
    var_12 = module_1._dict(var_11, var_2)
    assert var_12 == "{'y': {'a': 1}, 'x': {'b': 2}}"



# Parsed testcases at query #3
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'y = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'y = (1, 2, 3)'

import isort.literal as module_0

def test_case_0():
    var_0 = "z = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "z = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_literal_eval_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = set()
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = module_1._set(var_0, var_2)
    assert var_3 == '{}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 1
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = module_1.ISortPrettyPrinter(var_2)
    var_4 = module_1._set(var_1, var_3)
    assert var_4 == '{1}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.Config()
    var_5 = module_1.ISortPrettyPrinter(var_4)
    var_6 = module_1._set(var_3, var_5)
    assert var_6 == '{1, 2, 3}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'banana'
    var_1 = 'apple'
    var_2 = 'cherry'
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.Config()
    var_5 = module_1.ISortPrettyPrinter(var_4)
    var_6 = module_1._set(var_3, var_5)
    assert var_6 == "{'apple', 'banana', 'cherry'}"



# Parsed testcases at query #7
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]\n'

import isort.literal as module_0

def test_case_0():
    var_0 = "x = {'b': 2, 'a': 1}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "x = {'a': 1, 'b': 2}\n"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_2, var_0]
    var_4 = 80
    var_5 = module_0.Config()
    var_6 = module_1.ISortPrettyPrinter(var_5)
    var_7 = module_1._unique_list(var_3, var_6)
    assert var_7 == '[1, 2, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = []
    var_1 = 80
    var_2 = module_0.Config()
    var_3 = module_1.ISortPrettyPrinter(var_2)
    var_4 = module_1._unique_list(var_0, var_3)
    assert var_4 == '[]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 80
    var_3 = module_0.Config()
    var_4 = module_1.ISortPrettyPrinter(var_3)
    var_5 = module_1._unique_list(var_1, var_4)
    assert var_5 == '[5]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 80
    var_5 = module_0.Config()
    var_6 = module_1.ISortPrettyPrinter(var_5)
    var_7 = module_1._unique_list(var_3, var_6)
    assert var_7 == '[1, 2, 3]'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 3
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_1, var_0]
    var_4 = 80
    var_5 = module_0.Config()
    var_6 = module_1.ISortPrettyPrinter(var_5)
    var_7 = module_1._unique_list(var_3, var_6)
    assert var_7 == "[2, 3, 'a']"



# Parsed testcases at query #9
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'y = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'y = (1, 2, 3)'

import isort.literal as module_0

def test_case_0():
    var_0 = "z = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "z = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_literal_parsing_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = (var_0, var_1, var_2, var_2, var_0)
    var_4 = module_0.Config()
    var_5 = module_1.ISortPrettyPrinter(var_4)
    var_6 = module_1._unique_tuple(var_3, var_5)
    assert var_6 == '(1, 2, 3)'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = ()
    var_1 = module_0.Config()
    var_2 = module_1.ISortPrettyPrinter(var_1)
    var_3 = module_1._unique_tuple(var_0, var_2)
    assert var_3 == '()'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 42
    var_1 = (var_0,)
    var_2 = module_0.Config()
    var_3 = module_1.ISortPrettyPrinter(var_2)
    var_4 = module_1._unique_tuple(var_1, var_3)
    assert var_4 == '(42,)'



# Parsed testcases at query #12
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'z = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid code'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 100
    var_1 = lambda x, ext, cfg: x.upper()
    var_2 = module_0.Config()
    var_3 = 'x = [3, 1, 2]'
    var_4 = 'list'
    var_5 = '.py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)
    assert var_6 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = lambda x, y, z: x
    var_4 = module_0.Config()
    var_5 = module_1.assignment(var_0, var_1, var_2, var_4)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #14
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = lambda x, y, z: x
    var_4 = module_0.Config()
    var_5 = module_1.assignment(var_0, var_1, var_2, var_4)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #16
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "x = {'b': 2, 'a': 1}"
    var_1 = 'dictionary'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "x = {'a': 1, 'b': 2}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = {3, 1, 2}'
    var_1 = 'set'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = {1, 2, 3}'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = (1, 2, 3)'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'dictionary'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, _, __: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'
    var_3 = module_0.Config()
    var_4 = module_1.assignment(var_0, var_1, var_2, var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_literal_eval_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()



# Parsed testcases at query #21
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = (1, 2, 3)'

import isort.literal as module_0

def test_case_0():
    var_0 = 'y = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'y = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "z = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "z = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = (1, 2, 3)'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, ext, cfg: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'



# Parsed testcases at query #22
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, ext, cfg: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'z = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'w = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = "v = {'a': 1, 'b': 2}"
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #3
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
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'z = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'invalid = {1, 2, 3'
    var_1 = 'set'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, ext, cfg: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'



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
    var_6 = (var_3, var_4, var_5)
    var_7 = module_1._tuple(var_6, var_2)
    assert var_7 == '(1, 2, 3)'
    var_8 = 'b'
    var_9 = 'a'
    var_10 = 'c'
    var_11 = (var_8, var_9, var_10)
    var_12 = module_1._tuple(var_11, var_2)
    assert var_12 == "('a', 'b', 'c')"
    var_13 = (var_4,)
    var_14 = module_1._tuple(var_13, var_2)
    assert var_14 == '(1,)'
    var_15 = ()
    var_16 = module_1._tuple(var_15, var_2)
    assert var_16 == '()'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = set()
    var_3 = module_1._set(var_2, var_1)
    assert var_3 == '{}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 1
    var_3 = {var_2}
    var_4 = module_1._set(var_3, var_1)
    assert var_4 == '{1}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 3
    var_3 = 1
    var_4 = 2
    var_5 = {var_2, var_3, var_4}
    var_6 = module_1._set(var_5, var_1)
    assert var_6 == '{1, 2, 3}'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 'banana'
    var_3 = 'apple'
    var_4 = 'cherry'
    var_5 = {var_2, var_3, var_4}
    var_6 = module_1._set(var_5, var_1)
    assert var_6 == "{'apple', 'banana', 'cherry'}"



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_formatting_function_applied_when_configured. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = '_formatted'
    var_1 = lambda x, y, z: x + var_0
    var_2 = module_0.Config()
    var_3 = 'x = [3, 1, 2]'
    var_4 = 'list'
    var_5 = 'py'
    var_6 = module_1.assignment(var_3, var_4, var_5, var_2)



# Parsed testcases at query #7
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'z = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'w = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'v = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = ()
    var_3 = module_1._unique_tuple(var_2, var_1)
    assert var_3 == '()'

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 1
    var_3 = (var_2,)
    var_4 = module_1._unique_tuple(var_3, var_1)
    assert var_4 == '(1,)'
    var_5 = 'a'
    var_6 = (var_5,)
    var_7 = module_1._unique_tuple(var_6, var_1)
    assert var_7 == "('a',)"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 3
    var_3 = 1
    var_4 = 2
    var_5 = (var_2, var_3, var_4)
    var_6 = module_1._unique_tuple(var_5, var_1)
    assert var_6 == '(1, 2, 3)'
    var_7 = 'b'
    var_8 = 'a'
    var_9 = 'c'
    var_10 = (var_7, var_8, var_9)
    var_11 = module_1._unique_tuple(var_10, var_1)
    assert var_11 == "('a', 'b', 'c')"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 2
    var_3 = 1
    var_4 = 3
    var_5 = (var_2, var_2, var_3, var_4, var_4)
    var_6 = module_1._unique_tuple(var_5, var_1)
    assert var_6 == '(1, 2, 3)'
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 'z'
    var_10 = (var_7, var_8, var_7, var_9)
    var_11 = module_1._unique_tuple(var_10, var_1)
    assert var_11 == "('x', 'y', 'z')"

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1.ISortPrettyPrinter(var_0)
    var_2 = 3
    var_3 = 'a'
    var_4 = 1
    var_5 = 'b'
    var_6 = (var_2, var_3, var_4, var_5)
    var_7 = module_1._unique_tuple(var_6, var_1)
    assert var_7 == "(1, 3, 'a', 'b')"



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_literal_parsing_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = 'py'



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_2, var_0]
    var_4 = 80
    var_5 = module_0.Config()
    var_6 = module_1.ISortPrettyPrinter(var_5)
    var_7 = module_1._unique_list(var_3, var_6)
    assert var_7 == '[1, 2, 3]'



# Parsed testcases at query #12
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'y = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'y = (1, 2, 3)'

import isort.literal as module_0

def test_case_0():
    var_0 = "z = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "z = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, _, __: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'

import isort.literal as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == ''

import isort.literal as module_0

def test_case_0():
    var_0 = '   \n   \n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == ''



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #14
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "x = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "x = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, ext, cfg: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #15
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'z = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'w = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'v = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'u = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'U = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 't = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 't = [1, 2, 3]   \n'



# Parsed testcases at query #16
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "y = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "y = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda code, ext, cfg: code.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #18
#--------------------------




import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = "d = {'c': 3, 'a': 1, 'b': 2}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == "d = {'a': 1, 'b': 2, 'c': 3}"

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]'
    var_1 = 'invalid_type'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = invalid_literal'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)

import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, ext, cfg: x.upper()
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = '.py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'X = [1, 2, 3]'

import isort.literal as module_0

def test_case_0():
    var_0 = 'x = [3, 1, 2]   \n'
    var_1 = 'list'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.literal as module_1

def test_case_0():
    var_0 = lambda x, y, z: x
    var_1 = module_0.Config()
    var_2 = 'x = [3, 1, 2]'
    var_3 = 'list'
    var_4 = 'py'
    var_5 = module_1.assignment(var_2, var_3, var_4, var_1)
    assert var_5 == 'x = [1, 2, 3]'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_literal_eval_failure. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x = {1, 2, 3'
    var_1 = 'tuples'
    var_2 = 'py'



