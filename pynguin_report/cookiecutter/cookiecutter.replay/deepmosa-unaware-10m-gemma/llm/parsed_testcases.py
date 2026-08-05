####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'my_template.json'
    var_4 = module_0.get_file_name(var_0, var_3)
    var_5 = 'test'
    var_6 = 'test.json'
    var_7 = 'dir'
    var_8 = ''
    var_9 = module_0.get_file_name(var_7, var_8)
    var_10 = '.json'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = module_0.get_file_name(var_0, var_1)
    var_4 = 'my_template.json'
    var_5 = '/tmp/replay/my_template.json'
    var_6 = module_0.get_file_name(var_0, var_4)
    var_7 = '/tmp/replay'
    var_8 = 'test'
    var_9 = ''
    var_10 = module_0.get_file_name(var_0, var_9)
    assert var_10 == '/tmp/replay/.json'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'extra_data'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'tester'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 123
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = 'my_template.json'
    var_12 = 'already_has.json'
    var_13 = 'not_cookiecutter'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = 'invalid_template'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = 'nested_test'



# Parsed testcases at query #5
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
    var_18 = 'oops'
    var_19 = {var_17: var_18}
    var_20 = module_1.dump(var_0, var_1, var_19)
    var_21 = 'test.json'
    var_22 = '/tmp/replay/test.json'
    var_23 = module_1.dump(var_0, var_21, var_8)
    var_24 = 'w'
    var_25 = 'utf-8'



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
    var_14 = 'not_cookiecutter'
    var_15 = {var_14: var_7}
    var_16 = module_0.dump(var_0, var_1, var_15)
    var_17 = 'my_template.json'
    var_18 = '/tmp/replay/my_template.json'
    var_19 = module_0.dump(var_0, var_17, var_8)
    var_20 = 'w'
    var_21 = 'utf-8'



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
    var_18 = 'oops'
    var_19 = {var_17: var_18}
    var_20 = module_1.dump(var_0, var_1, var_19)
    var_21 = 'my_template.json'
    var_22 = '/tmp/replay/my_template.json'
    var_23 = module_1.dump(var_0, var_21, var_8)
    var_24 = 'w'
    var_25 = 'utf-8'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'my_template'
    var_1 = 'replays'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'not_cookiecutter'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = 'missing_template'



# Parsed testcases at query #14
#--------------------------




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
    var_9 = 'my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = module_0.dump(var_0, var_14, var_8)
    var_16 = 'w'
    var_17 = 'utf-8'
    var_18 = 'not_cookiecutter'
    var_19 = {var_18: var_7}
    var_20 = module_0.dump(var_0, var_1, var_19)



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
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = '/tmp/replay/my_template.json'
    var_16 = module_0.dump(var_0, var_14, var_8)
    var_17 = 'w'
    var_18 = 'utf-8'
    var_19 = 'not_cookiecutter'
    var_20 = 'oops'
    var_21 = {var_19: var_20}
    var_22 = module_0.dump(var_0, var_1, var_21)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
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
    var_18 = {var_17: var_7}
    var_19 = module_0.dump(var_0, var_1, var_18)



# Parsed testcases at query #20
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
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = module_0.dump(var_0, var_14, var_8)
    var_16 = 'w'
    var_17 = 'utf-8'
    var_18 = 'not_cookiecutter'
    var_19 = {var_18: var_7}



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/replay/my_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'
    var_11 = 2
    var_12 = 'my_template.json'
    var_13 = '/tmp/replay/my_template.json'
    var_14 = module_0.dump(var_0, var_12, var_6)
    var_15 = 'w'
    var_16 = 'utf-8'
    var_17 = 'not_cookiecutter'
    var_18 = {}
    var_19 = {var_17: var_18}
    var_20 = module_0.dump(var_0, var_1, var_19)



# Parsed testcases at query #27
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
    var_18 = 'oops'
    var_19 = {var_17: var_18}
    var_20 = module_1.dump(var_0, var_1, var_19)
    var_21 = 'template.json'
    var_22 = '/tmp/replay/template.json'
    var_23 = module_1.dump(var_0, var_21, var_8)
    var_24 = 'w'
    var_25 = 'utf-8'



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




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
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
    var_20 = {}
    var_21 = {var_19: var_20}
    var_22 = module_0.dump(var_0, var_1, var_21)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'my_template'
    var_1 = 'cookiecutter'
    var_2 = 'other'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'replay'
    var_9 = 'my_template.json'
    var_10 = 'not_cookiecutter'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 'my_template.json'

def test_case_0():
    var_0 = 'test'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = 0



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
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



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
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
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = module_1.dump(var_0, var_1, var_19)
    var_21 = 'my_template.json'
    var_22 = '/tmp/replay/my_template.json'
    var_23 = module_1.dump(var_0, var_21, var_8)
    var_24 = 'w'
    var_25 = 'utf-8'



# Parsed testcases at query #15
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
    var_9 = 'my_template.json'
    var_10 = 2
    var_11 = module_0.dumps(var_8, indent=var_10)
    var_12 = module_1.dump(var_0, var_1, var_8)
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = ''
    var_16 = 0
    var_17 = 'not_cookiecutter'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = module_1.dump(var_0, var_1, var_19)
    var_21 = 'template.json'
    var_22 = 'template.json'
    var_23 = module_1.dump(var_0, var_21, var_8)
    var_24 = 'w'
    var_25 = 'utf-8'



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




# Parsed testcases at query #21
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'
    var_10 = 'my_template.json'
    var_11 = 'not_cookiecutter'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = {var_2: var_14}
    var_16 = '/fake/dir'
    var_17 = 'test'
    var_18 = module_0.dump(var_16, var_17, var_15)
    var_19 = 'test.json'
    var_20 = 'w'
    var_21 = 'utf-8'



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
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
    var_13 = 2
    var_14 = 'my_template.json'
    var_15 = module_0.dump(var_0, var_14, var_8)
    var_16 = 'w'
    var_17 = 'utf-8'
    var_18 = 'not_cookiecutter'
    var_19 = {var_18: var_7}
    var_20 = module_0.dump(var_0, var_1, var_19)



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
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



# Parsed testcases at query #31
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
    var_19 = {}
    var_20 = {var_18: var_19}
    var_21 = module_0.dump(var_0, var_1, var_20)



# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #35
#--------------------------




