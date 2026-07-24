####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'my_template.json'
    var_5 = '/tmp/template.json'
    var_6 = module_0.get_file_name(var_0, var_4)
    assert var_6 == '/tmp/replay/my_template.json'
    var_7 = '/tmp/replay'
    var_8 = 'test'
    var_9 = '/tmp/replay/test.json'
    var_10 = ''
    var_11 = module_0.get_file_name(var_0, var_10)
    assert var_11 == '/tmp/replay/.json'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'config.json'
    var_5 = 'config.json'
    var_6 = module_0.get_file_name(var_0, var_4)
    var_7 = '/tmp/replay'
    var_8 = 'test.json'
    var_9 = 'test'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'my_template.json'
    var_5 = '/tmp/regex/my_template.json'
    var_6 = '/regex/'
    var_7 = '/replay/'
    var_8 = '/tmp/replay/my_template.json'
    var_9 = module_0.get_file_name(var_0, var_4)
    var_10 = '/tmp/replay'
    var_11 = '/tmp/replay/test.json'
    var_12 = 'test'
    var_13 = ''
    var_14 = module_0.get_file_name(var_0, var_13)
    assert var_14 == '/tmp/replay/.json'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replays'
    var_1 = 'my_template'
    var_2 = '/tmp/replays/my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'config.json'
    var_5 = '/tmp/replays/config.json'
    var_6 = module_0.get_file_name(var_0, var_4)
    var_7 = '/tmp/replays'
    var_8 = '/tmp/replays/test.json'
    var_9 = 'test'
    var_10 = ''
    var_11 = module_0.get_file_name(var_0, var_10)
    assert var_11 == '/tmp/replays/.json'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'my_template.json'
    var_5 = '/tmp/lag/my_template.json'
    var_6 = module_0.get_file_name(var_0, var_4)
    assert var_6 == '/tmp/replay/my_template.json'
    var_7 = '/tmp/replay'
    var_8 = '/tmp/replay/other_template.json'
    var_9 = 'other_template'
    var_10 = '.json'
    var_11 = module_0.get_file_name(var_0, var_10)
    assert var_11 == '/tmp/replay/.json'



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'other_data'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'tester'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 123
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 'replays'
    var_11 = 'no_key'
    var_12 = 'here'
    var_13 = {var_11: var_12}
    var_14 = 'non_existent_template'



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = 2
    var_11 = module_0.dumps(var_8, indent=var_10)
    var_12 = module_1.dump(var_0, var_1, var_8)
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = 0
    var_16 = module_0.loads(var_10)
    var_17 = 'not_cookiecutter'
    var_18 = {var_17: var_15}
    var_19 = module_1.dump(var_0, var_1, var_18)
    var_20 = 'my_template.json'
    var_21 = '/tmp/replay/my_template.json'
    var_22 = module_1.dump(var_0, var_20, var_8)
    var_23 = 'w'
    var_24 = 'utf-8'



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = '/tmp/replay/my_template.json'
    var_16 = module_0.dump(var_0, var_14, var_8)
    var_17 = 'not_cookiecutter'
    var_18 = {}
    var_19 = {var_17: var_18}
    var_20 = module_0.dump(var_0, var_1, var_19)
    var_21 = {}
    var_22 = module_0.dump(var_0, var_1, var_21)



# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'w'
    var_1 = 'utf-8'
    var_2 = 2

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/replay'
    var_4 = 'template'
    var_5 = module_0.dump(var_3, var_4, var_2)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'w'
    var_1 = 'utf-8'
    var_2 = 2

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/replay'
    var_4 = 'template'
    var_5 = module_0.dump(var_3, var_4, var_2)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 123
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'
    var_10 = 'my_template.json'
    var_11 = 'not_cookiecutter'
    var_12 = {}
    var_13 = {var_11: var_12}



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'w'
    var_1 = 'utf-8'
    var_2 = 'dump'
    var_3 = None
    var_4 = (var_3, var_3)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'w'
    var_1 = 'utf-8'
    var_2 = 2

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/replays'
    var_4 = 'test'
    var_5 = module_0.dump(var_3, var_4, var_2)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 'my_template.json'
    var_14 = '/tmp/replay/my_template.json'
    var_15 = module_0.dump(var_0, var_13, var_8)
    var_16 = 'not_cookiecutter'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = module_0.dump(var_0, var_1, var_18)



# Parsed testcases at query #12
#--------------------------


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = 2
    var_11 = module_0.dumps(var_8, indent=var_10)
    var_12 = module_1.dump(var_0, var_1, var_8)
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = ''
    var_16 = 0
    var_17 = 'not_cookiecutter'
    var_18 = {var_17: var_15}
    var_19 = module_1.dump(var_0, var_1, var_18)
    var_20 = 'my_template.json'
    var_21 = '/tmp/replay/my_template.json'
    var_22 = module_1.dump(var_0, var_20, var_8)
    var_23 = 'w'
    var_24 = 'utf-8'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = '/tmp/replay/my_template.json'
    var_16 = module_0.dump(var_0, var_14, var_8)
    var_17 = 'w'
    var_18 = 'utf-8'
    var_19 = 'not_cookiecutter'
    var_20 = {var_19: var_7}
    var_21 = module_0.dump(var_0, var_1, var_20)



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 'my_template.json'
    var_14 = '/tmp/replay/my_template.json'
    var_15 = module_0.dump(var_0, var_13, var_8)
    var_16 = 'w'
    var_17 = 'utf-8'
    var_18 = 'not_cookiecutter'
    var_19 = {var_18: var_7}
    var_20 = module_0.dump(var_0, var_1, var_19)
    var_21 = '/tmp/replay'
    var_22 = 0
    var_23 = str(var_12)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 'my_template.json'
    var_14 = '/tmp/replay/my_template.json'
    var_15 = module_0.dump(var_0, var_13, var_8)
    var_16 = 'w'
    var_17 = 'utf-8'
    var_18 = 'not_cookiecutter'
    var_19 = {var_18: var_7}
    var_20 = module_0.dump(var_0, var_1, var_19)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = '/tmp/replay/my_template.json'
    var_16 = module_0.dump(var_0, var_14, var_8)
    var_17 = 'w'
    var_18 = 'utf-8'
    var_19 = 'not_cookiecutter'
    var_20 = {var_19: var_7}
    var_21 = module_0.dump(var_0, var_1, var_20)



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'other'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'data'
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = 'not_cookiecutter'
    var_11 = {var_10: var_8}
    var_12 = 'missing_template'
    var_13 = 'my_template.json'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '/tmp/replay/my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = '/tmp/replay/my_template.json'
    var_16 = module_0.dump(var_0, var_14, var_8)
    var_17 = 'w'
    var_18 = 'utf-8'
    var_19 = 'not_cookiecutter'
    var_20 = {var_19: var_7}
    var_21 = module_0.dump(var_0, var_1, var_20)



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'other'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'data'
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = 'not_cookiecutter'
    var_11 = True
    var_12 = {var_10: var_11}
    var_13 = 'does_not_exist.json'
    var_14 = 'does_not_exist'
    var_15 = 'template.json'
    var_16 = 'template.json'



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #35
#--------------------------




# Parsed testcases at query #36
#--------------------------




# Parsed testcases at query #37
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 'my_template.json'
    var_14 = module_0.dump(var_0, var_13, var_8)
    var_15 = 'w'
    var_16 = 'utf-8'
    var_17 = 'not_cookiecutter'
    var_18 = {var_17: var_7}



