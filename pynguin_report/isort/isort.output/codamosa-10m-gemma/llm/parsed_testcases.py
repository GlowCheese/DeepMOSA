####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = -1
    var_1 = 'stdlib'
    var_2 = 'Standard Library'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = 0
    var_6 = ''
    var_7 = [var_6]
    var_8 = '\n'
    var_9 = 'STDLIB'
    var_10 = [var_9]
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = {}
    var_14 = 'sys'
    var_15 = 'os'
    var_16 = 'from sys import path'
    var_17 = 'from os import *'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = {var_11: var_13, var_12: var_18}
    var_20 = {var_9: var_19}
    var_21 = {}
    var_22 = {}
    var_23 = module_0.ParsedContent()
    var_24 = 2
    var_25 = '\n\n'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'z_module'
    var_5 = 'a_module'
    var_6 = 'import z_module'
    var_7 = 'import a_module'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'sys_mod'
    var_10 = 'from sys import path'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_8, var_3: var_11}
    var_13 = 'requests'
    var_14 = 'import requests'
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_2: var_15, var_3: var_16}

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIM'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'a'
    var_5 = 'import a'
    var_6 = {var_4: var_5}
    var_7 = 'b'
    var_8 = 'from b import c'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = 'd'
    var_12 = 'import d'
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_2: var_13, var_3: var_14}

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = 'import os'
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = {var_4: var_11, var_5: var_12}

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = {}
    var_4 = 'module_a'
    var_5 = 'module_b'
    var_6 = 'from module_a import x'
    var_7 = 'from module_b import *'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_1: var_3, var_2: var_8}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'z_module'
    var_5 = 'a_module'
    var_6 = ''
    var_7 = {var_4: var_6, var_5: var_6}
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = 'b_module'
    var_11 = {var_10: var_6}
    var_12 = {}
    var_13 = {var_2: var_11, var_3: var_12}
    var_14 = '# Header'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = 'os'
    var_7 = ''
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = {}
    var_12 = {}
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = '# Existing'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'a'
    var_5 = ''
    var_6 = {var_4: var_5}
    var_7 = 'b'
    var_8 = {var_7: var_5}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = 'c'
    var_11 = {var_10: var_5}
    var_12 = 'd'
    var_13 = {var_12: var_5}
    var_14 = {var_2: var_11, var_3: var_13}

import re as module_0

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRPTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'straight_mod'
    var_5 = ''
    var_6 = {var_4: var_5}
    var_7 = 'from_mod'
    var_8 = {var_7: var_5}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = '\n'
    var_14 = module_0.split(var_13)
    var_15 = enumerate(var_14)
    var_16 = 'from_mod'
    var_17 = next(var_3)
    var_18 = enumerate(var_14)
    var_19 = 'straight_mod'
    var_20 = next(var_6)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'from sys'
    var_1 = 'import os'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRD_PARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'import os'
    var_6 = {var_4: var_5}
    var_7 = 'sys'
    var_8 = 'from sys import path'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = 'requests'
    var_12 = 'import requests'
    var_13 = {var_11: var_12}
    var_14 = 'json'
    var_15 = 'from json import dumps'
    var_16 = {var_14: var_15}
    var_17 = {var_2: var_13, var_3: var_16}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = []



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'sys'
    var_1 = 'os'
    var_2 = 'import sys'
    var_3 = 'import os'
    var_4 = 'straight'
    var_5 = 'from'
    var_6 = {}
    var_7 = {}

def test_case_0():
    var_0 = 'os'
    var_1 = 'import os'
    var_2 = 'sys'
    var_3 = 'from sys import path'
    var_4 = 'from sys import path'
    var_5 = 'import os'

def test_case_0():
    var_0 = 'straight'
    var_1 = 'from'
    var_2 = 'os'
    var_3 = 'import os'
    var_4 = {var_2: var_3}
    var_5 = {}

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'Standard Library'
    var_2 = 'os'
    var_3 = 'import os'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'standard'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = 'z_module'
    var_4 = 'a_module'
    var_5 = 'import z_module'
    var_6 = 'import a_module'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'sys_module'
    var_9 = 'from sys import path'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_7, var_2: var_10}

def test_case_0():
    var_0 = 'standard'
    var_1 = 'FUTURE'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'import os'
    var_6 = {var_4: var_5}
    var_7 = 'sys'
    var_8 = 'from sys import path'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = '__future__'
    var_12 = 'from __future__ import annotations'
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_2: var_13, var_3: var_14}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = 'sys'
    var_4 = 'os'
    var_5 = 'import sys'
    var_6 = 'import os'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {}
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'x = 1'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = 'os'
    var_4 = 'import os'
    var_5 = {var_3: var_4}
    var_6 = 'sys'
    var_7 = 'from sys import argv'
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 'from sys import argv'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRD_PARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'import os'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'requests'
    var_10 = 'import requests'
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = {var_2: var_11, var_3: var_12}

def test_case_0():
    var_0 = 'stdlib'
    var_1 = 'Standard Library'
    var_2 = 'STDLIB'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'os'
    var_6 = 'import os'
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'STDLIB'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'import os'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = False



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'sys'
    var_5 = 'os'
    var_6 = 'import sys'
    var_7 = 'import os'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'math'
    var_10 = 'from math import sqrt'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_8, var_3: var_11}
    var_13 = 'requests'
    var_14 = 'import requests'
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = 'x = 1'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'FUTURE'
    var_3 = 'straight'
    var_4 = 'from'
    var_5 = 'a'
    var_6 = 'import a'
    var_7 = {var_5: var_6}
    var_8 = 'b'
    var_9 = 'from b import c'
    var_10 = {var_8: var_9}
    var_11 = {var_3: var_7, var_4: var_10}
    var_12 = 'd'
    var_13 = 'import d'
    var_14 = {var_12: var_13}
    var_15 = 'e'
    var_16 = 'from e import f'
    var_17 = {var_15: var_16}
    var_18 = {var_3: var_14, var_4: var_17}
    var_19 = {}
    var_20 = {}
    var_21 = {var_3: var_19, var_4: var_20}

def test_case_0():
    var_0 = 'def foo(): pass'
    var_1 = False
    var_2 = ''
    var_3 = []
    var_4 = 'pyi'
    var_5 = '\n'



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import isort.output as module_2

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRD_PARTY'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = 'import os'
    var_7 = 'import sys'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'collections import Counter'
    var_10 = 'from collections import Counter'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_8, var_3: var_11}
    var_13 = 'requests'
    var_14 = 'import requests'
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = {var_0: var_12, var_1: var_17}
    var_19 = 0
    var_20 = "print('hello')"
    var_21 = [var_20]
    var_22 = '\n'
    var_23 = [var_0, var_1]
    var_24 = {}
    var_25 = {}
    var_26 = 1
    var_27 = module_0.ParsedContent()
    var_28 = []
    var_29 = []
    var_30 = False
    var_31 = False
    var_32 = False
    var_33 = False
    var_34 = False
    var_35 = {}
    var_36 = {}
    var_37 = True
    var_38 = []
    var_39 = False
    var_40 = False
    var_41 = None
    var_42 = -1
    var_43 = 'default'
    var_44 = []
    var_45 = module_1.Config()
    var_46 = module_2.sorted_imports(var_27, var_45)
    var_47 = -1
    var_48 = "print('no imports')"
    var_49 = [var_48]
    var_50 = []
    var_51 = {}
    var_52 = {}
    var_53 = {}
    var_54 = module_0.ParsedContent()
    var_55 = module_2.sorted_imports(var_54, var_45)
    assert var_55 == "print('no imports')"
    var_56 = []
    var_57 = []
    var_58 = True
    var_59 = False
    var_60 = False
    var_61 = False
    var_62 = False
    var_63 = {}
    var_64 = {}
    var_65 = True
    var_66 = []
    var_67 = False
    var_68 = False
    var_69 = -1
    var_70 = []
    var_71 = module_1.Config()
    var_72 = module_2.sorted_imports(var_27, var_71)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = "print('hello')"
    var_1 = [var_0]
    var_2 = {}
    var_3 = -1
    var_4 = '\n'
    var_5 = 'standard'
    var_6 = [var_5]
    var_7 = [var_0]
    var_8 = 'straight'
    var_9 = 'from'
    var_10 = 'os'
    var_11 = 'sys'
    var_12 = 'import os'
    var_13 = 'import sys'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'math'
    var_16 = 'from math import sqrt'
    var_17 = {var_15: var_16}
    var_18 = {var_8: var_14, var_9: var_17}
    var_19 = {var_5: var_18}
    var_20 = 0
    var_21 = [var_5]
    var_22 = True
    var_23 = [var_0]
    var_24 = {}
    var_25 = 'from os import path'
    var_26 = 'from sys import *'
    var_27 = {var_10: var_25, var_11: var_26}
    var_28 = {var_8: var_24, var_9: var_27}
    var_29 = {var_5: var_28}
    var_30 = [var_5]
    var_31 = 'from sys import *'
    var_32 = 'from os import path'
    var_33 = [var_31]
    var_34 = 'third_party'
    var_35 = 'FUTURE'
    var_36 = 'a'
    var_37 = 'import a'
    var_38 = {var_36: var_37}
    var_39 = 'b'
    var_40 = 'from b import c'
    var_41 = {var_39: var_40}
    var_42 = {var_8: var_38, var_9: var_41}
    var_43 = 'd'
    var_44 = 'import d'
    var_45 = {var_43: var_44}
    var_46 = 'e'
    var_47 = 'from e import f'
    var_48 = {var_46: var_47}
    var_49 = {var_8: var_45, var_9: var_48}
    var_50 = 'f'
    var_51 = 'from __future__ import annotations'
    var_52 = {var_50: var_51}
    var_53 = {}
    var_54 = {var_8: var_52, var_9: var_53}
    var_55 = {var_5: var_42, var_34: var_49, var_35: var_54}
    var_56 = [var_5, var_34]



