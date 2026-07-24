####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'F401'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    var_9 = module_0.ignore_line(var_7, var_3, var_0)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = [var_2]
    var_4 = module_0.parse_noqa(var_3)
    var_5 = 'import sys  # noqa: E123, W451'
    var_6 = [var_5]
    var_7 = module_0.parse_noqa(var_6)
    var_8 = 'import os  # NOQA: F401'
    var_9 = [var_8]
    var_10 = module_0.parse_noqa(var_9)
    var_11 = 'import os  # noqa: E123'
    var_12 = 'import sys  # noqa'
    var_13 = 'import math  # noqa: F841'
    var_14 = "print('hello')  # no error"
    var_15 = [var_11, var_12, var_13, var_14]
    var_16 = module_0.parse_noqa(var_15)
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = 'import os  # noqa: E123,   E456,F789'
    var_19 = [var_18]
    var_20 = module_0.parse_noqa(var_19)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 'V104'
    var_3 = 1
    var_4 = {var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = 3
    var_8 = {var_7}
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = module_0.ignore_line(var_9, var_3, var_0)
    assert var_10 is True
    var_11 = 'E501'
    var_12 = module_0.ignore_line(var_9, var_5, var_11)
    assert var_12 is True
    var_13 = module_0.ignore_line(var_9, var_7, var_0)
    assert var_13 is False
    var_14 = 4
    var_15 = module_0.ignore_line(var_9, var_14, var_0)
    assert var_15 is False



# Parsed testcases at query #3
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = 'import sys'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_noqa(var_4)
    var_6 = 'import os  # noqa: E123'
    var_7 = 'import sys  # noqa: W451, F921'
    var_8 = [var_6, var_7]
    var_9 = module_0.parse_noqa(var_8)
    var_10 = 'import os  # noqa: F401'
    var_11 = [var_10]
    var_12 = module_0.parse_noqa(var_11)
    var_13 = 'import os  # NoQA: E123'
    var_14 = 'import sys  # noqa:  E456 , F888 '
    var_15 = [var_13, var_14]
    var_16 = module_0.parse_noqa(var_15)
    var_17 = 'import os'
    var_18 = [var_17, var_3]
    var_19 = module_0.parse_noqa(var_18)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'V104'
    var_2 = 'all'
    var_3 = 1
    var_4 = {var_3}
    var_5 = 5
    var_6 = {var_5}
    var_7 = 10
    var_8 = {var_7}
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = module_0.ignore_line(var_9, var_3, var_0)
    assert var_10 is True
    var_11 = 'E456'
    var_12 = module_0.ignore_line(var_9, var_7, var_11)
    assert var_12 is True
    var_13 = 2
    var_14 = module_0.ignore_line(var_9, var_13, var_0)
    assert var_14 is False
    var_15 = module_0.ignore_line(var_9, var_5, var_0)
    assert var_15 is False



# Parsed testcases at query #4
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'F401'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    var_9 = module_0.ignore_line(var_7, var_3, var_0)
    assert var_9 is True
    var_10 = module_0.ignore_line(var_7, var_5, var_1)
    assert var_10 is True
    var_11 = 'all'
    var_12 = 7
    var_13 = 8
    var_14 = {var_12, var_13}
    var_15 = {var_2}
    var_16 = {var_11: var_14, var_0: var_15}
    var_17 = module_0.ignore_line(var_16, var_12, var_0)
    assert var_17 is True
    var_18 = module_0.ignore_line(var_16, var_13, var_1)
    assert var_18 is True
    var_19 = {var_2}
    var_20 = 1
    var_21 = {var_20}
    var_22 = {var_0: var_19, var_11: var_21}
    var_23 = module_0.ignore_line(var_22, var_20, var_0)
    assert var_23 is True
    var_24 = {var_2}
    var_25 = {var_0: var_24}
    var_26 = 6
    var_27 = module_0.ignore_line(var_25, var_26, var_0)
    assert var_27 is False
    var_28 = module_0.ignore_line(var_25, var_2, var_1)
    assert var_28 is False
    var_29 = {}
    var_30 = module_0.ignore_line(var_29, var_20, var_0)
    assert var_30 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os  # noqa: F401'
    var_1 = 'x = 1  # noqa'
    var_2 = 'print(x)  # noqa: E123, W451'
    var_3 = 'y = 2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_noqa(var_4)

import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa: F841, E501'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)



# Parsed testcases at query #5
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = [var_2]
    var_4 = module_0.parse_noqa(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'import sys  # noqa: E123, W451'
    var_7 = [var_6]
    var_8 = module_0.parse_noqa(var_7)
    var_9 = 'import os  # noqa: F401, F841'
    var_10 = [var_9]
    var_11 = module_0.parse_noqa(var_10)
    var_12 = 'import os  # NOQA: E123'
    var_13 = 'import sys  # noqa:   E461, E701 '
    var_14 = [var_12, var_13]
    var_15 = module_0.parse_noqa(var_14)
    var_16 = 'import os  # noqa: E123'
    var_17 = 'import sys  # noqa'
    var_18 = 'import math # noqa: F401'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.parse_noqa(var_19)
    var_21 = 'import os'
    var_22 = 'import sys'
    var_23 = [var_21, var_22]
    var_24 = module_0.parse_noqa(var_23)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 'V104'
    var_3 = 1
    var_4 = {var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = 3
    var_8 = {var_7}
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = module_0.ignore_line(var_9, var_3, var_0)
    assert var_10 is True
    var_11 = 'E501'
    var_12 = module_0.ignore_line(var_9, var_5, var_11)
    assert var_12 is True
    var_13 = 4
    var_14 = module_0.ignore_line(var_9, var_13, var_0)
    assert var_14 is False
    var_15 = 'F401'
    var_16 = module_0.ignore_line(var_9, var_7, var_15)
    assert var_16 is True



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'F401'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    var_9 = module_0.ignore_line(var_7, var_3, var_0)
    assert var_9 is True
    var_10 = 7
    var_11 = module_0.ignore_line(var_7, var_10, var_0)
    assert var_11 is False
    var_12 = 'all'
    var_13 = 1
    var_14 = 3
    var_15 = {var_13, var_14}
    var_16 = {var_2}
    var_17 = {var_12: var_15, var_0: var_16}
    var_18 = module_0.ignore_line(var_17, var_13, var_0)
    assert var_18 is True
    var_19 = 'F888'
    var_20 = module_0.ignore_line(var_17, var_14, var_19)
    assert var_20 is True
    var_21 = module_0.ignore_line(var_17, var_2, var_0)
    assert var_21 is True
    var_22 = module_0.ignore_line(var_17, var_5, var_0)
    assert var_22 is False
    var_23 = {var_2}
    var_24 = {var_0: var_23}
    var_25 = 6
    var_26 = module_0.ignore_line(var_24, var_25, var_0)
    assert var_26 is False
    var_27 = module_0.ignore_line(var_24, var_2, var_1)
    assert var_27 is False
    var_28 = {}
    var_29 = module_0.ignore_line(var_28, var_13, var_0)
    assert var_29 is False
    var_30 = 'import os  # noqa'
    var_31 = 'import sys  # noqa: F401'
    var_32 = 'x = 1  # noqa: E123, W451'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.parse_noqa(var_33)
    var_35 = module_0.ignore_line(var_34, var_13, var_0)
    assert var_35 is True
    var_36 = 'V104'
    var_37 = module_0.ignore_line(var_34, var_5, var_36)
    assert var_37 is True
    var_38 = module_0.ignore_line(var_34, var_5, var_1)
    assert var_38 is False
    var_39 = module_0.ignore_line(var_34, var_14, var_0)
    assert var_39 is True
    var_40 = 'W451'
    var_41 = module_0.ignore_line(var_34, var_14, var_40)
    assert var_41 is True



# Parsed testcases at query #2
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'F401'
    var_2 = 'all'
    var_3 = 5
    var_4 = 10
    var_5 = {var_3, var_4}
    var_6 = 2
    var_7 = {var_6}
    var_8 = 7
    var_9 = {var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = module_0.ignore_line(var_10, var_3, var_0)
    assert var_11 is True
    var_12 = module_0.ignore_line(var_10, var_4, var_0)
    assert var_12 is True
    var_13 = 'E501'
    var_14 = module_0.ignore_line(var_10, var_8, var_13)
    assert var_14 is True
    var_15 = module_0.ignore_line(var_10, var_6, var_1)
    assert var_15 is True
    var_16 = 1
    var_17 = module_0.ignore_line(var_10, var_16, var_0)
    assert var_17 is False
    var_18 = 8
    var_19 = module_0.ignore_line(var_10, var_18, var_0)
    assert var_19 is False
    var_20 = 9
    var_21 = module_0.ignore_line(var_10, var_20, var_1)
    assert var_21 is False
    var_22 = {}
    var_23 = module_0.ignore_line(var_22, var_16, var_0)
    assert var_23 is False
    var_24 = {var_16}
    var_25 = {var_0: var_24}
    var_26 = module_0.ignore_line(var_25, var_6, var_0)
    assert var_26 is False



# Parsed testcases at query #3
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = 'import sys'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_noqa(var_4)
    var_6 = 'import os  # noqa: E123'
    var_7 = [var_6, var_3]
    var_8 = module_0.parse_noqa(var_7)
    var_9 = 'import os  # noqa: E123, W451, F921'
    var_10 = 'import sys  # noqa:E722'
    var_11 = [var_9, var_10]
    var_12 = module_0.parse_noqa(var_11)
    var_13 = 'import os  # NOQA: F401'
    var_14 = 'import sys  # noqa: F841'
    var_15 = [var_13, var_14]
    var_16 = module_0.parse_noqa(var_15)
    var_17 = 'import sys  # noqa: E123'
    var_18 = [var_2, var_17]
    var_19 = module_0.parse_noqa(var_18)
    var_20 = 'import os'
    var_21 = [var_20, var_3]
    var_22 = module_0.parse_noqa(var_21)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 1
    var_3 = {var_2}
    var_4 = 2
    var_5 = {var_4}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_0.ignore_line(var_6, var_2, var_0)
    assert var_7 is True
    var_8 = 'F401'
    var_9 = module_0.ignore_line(var_6, var_4, var_8)
    assert var_9 is True
    var_10 = 3
    var_11 = module_0.ignore_line(var_6, var_10, var_0)
    assert var_11 is False
    var_12 = 5
    var_13 = 'E999'
    var_14 = module_0.ignore_line(var_6, var_12, var_13)
    assert var_14 is False



# Parsed testcases at query #4
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = 'import sys'
    var_4 = [var_2, var_3]
    var_5 = 'all'
    var_6 = 1
    var_7 = {var_6}
    var_8 = {var_5: var_7}
    var_9 = module_0.parse_noqa(var_4)
    var_10 = 'import os  # noqa: E123'
    var_11 = 'import sys  # NOQA: W451'
    var_12 = 'import math  # noqa: F401, E501'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'E123'
    var_15 = 'W451'
    var_16 = 'V104'
    var_17 = 'E501'
    var_18 = {var_6}
    var_19 = 2
    var_20 = {var_19}
    var_21 = 3
    var_22 = {var_21}
    var_23 = {var_21}
    var_24 = {var_14: var_18, var_15: var_20, var_16: var_22, var_17: var_23}
    var_25 = module_0.parse_noqa(var_13)
    var_26 = 'x = 1  # noqa: E123,  W451, F841'
    var_27 = [var_26]
    var_28 = 'V107'
    var_29 = {var_6}
    var_30 = {var_6}
    var_31 = {var_6}
    var_32 = {var_14: var_29, var_15: var_30, var_28: var_31}
    var_33 = module_0.parse_noqa(var_27)
    var_34 = 'import a  # noqa: E123'
    var_35 = 'import b  # noqa: E123'
    var_36 = 'import c'
    var_37 = [var_34, var_35, var_36]
    var_38 = {var_6, var_19}
    var_39 = {var_14: var_38}
    var_40 = module_0.parse_noqa(var_37)
    var_41 = "print('hello')"
    var_42 = 'x = 10'
    var_43 = [var_41, var_42]
    var_44 = module_0.parse_noqa(var_43)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 'V104'
    var_3 = 1
    var_4 = {var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = 3
    var_8 = {var_7}
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = module_0.ignore_line(var_9, var_3, var_0)
    assert var_10 is True
    var_11 = 'F401'
    var_12 = module_0.ignore_line(var_9, var_5, var_11)
    assert var_12 is True
    var_13 = module_0.ignore_line(var_9, var_7, var_0)
    assert var_13 is False
    var_14 = 4
    var_15 = module_0.ignore_line(var_9, var_14, var_0)
    assert var_15 is False



# Parsed testcases at query #5
#--------------------------


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'import os  # noqa'
    var_3 = 'import sys  # NoQA'
    var_4 = [var_2, var_3]
    var_5 = 'all'
    var_6 = 1
    var_7 = 2
    var_8 = {var_6, var_7}
    var_9 = {var_5: var_8}
    var_10 = module_0.parse_noqa(var_4)
    var_11 = 'import os  # noqa: E123'
    var_12 = 'import sys  # noqa: E123, W451'
    var_13 = [var_11, var_12]
    var_14 = 'E123'
    var_15 = 'W451'
    var_16 = {var_6, var_7}
    var_17 = {var_7}
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.parse_noqa(var_13)
    var_20 = 'import os  # noqa: F401'
    var_21 = 'x = 1     # noqa: F841'
    var_22 = [var_20, var_21]
    var_23 = 'V104'
    var_24 = 'V107'
    var_25 = {var_6}
    var_26 = {var_7}
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = module_0.parse_noqa(var_22)
    var_29 = 'import os'
    var_30 = 'import sys  # noqa: E123'
    var_31 = 'import math  # noqa'
    var_32 = 'import re    # noqa: F401'
    var_33 = [var_29, var_30, var_31, var_32]
    var_34 = {var_7}
    var_35 = 3
    var_36 = {var_35}
    var_37 = 4
    var_38 = {var_37}
    var_39 = {var_14: var_34, var_5: var_36, var_23: var_38}
    var_40 = module_0.parse_noqa(var_33)
    var_41 = 'import os  # NOQA: E123,   E456'
    var_42 = [var_41]
    var_43 = 'E456'
    var_44 = {var_6}
    var_45 = {var_6}
    var_46 = {var_14: var_44, var_43: var_45}
    var_47 = module_0.parse_noqa(var_42)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 'V104'
    var_3 = 1
    var_4 = {var_3}
    var_5 = 2
    var_6 = {var_5}
    var_7 = 3
    var_8 = {var_7}
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = module_0.ignore_line(var_9, var_3, var_0)
    assert var_10 is True
    var_11 = 'E501'
    var_12 = module_0.ignore_line(var_9, var_5, var_11)
    assert var_12 is True
    var_13 = 4
    var_14 = module_0.ignore_line(var_9, var_13, var_0)
    assert var_14 is False
    var_15 = 5
    var_16 = module_0.ignore_line(var_9, var_15, var_2)
    assert var_16 is False



